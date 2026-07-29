# Reconhecimento Facial e Contagem de Pessoas

Conjunto de protótipos em Python para **identificar pessoas por reconhecimento facial** e **contar pessoas únicas** em vídeo, a partir de arquivos de vídeo, webcam ou de um celular usado como câmera IP (DroidCam).

O repositório reúne experimentos incrementais desenvolvidos no contexto do laboratório **LIA — UEFS**. Cada script é autocontido e representa uma etapa da evolução do estudo: começa na comparação de duas fotos e chega a um pipeline em tempo real que aprende rostos novos durante a execução.

---

## Sumário

- [Motivação e escopo](#motivação-e-escopo)
- [Visão geral da arquitetura](#visão-geral-da-arquitetura)
- [Scripts do repositório](#scripts-do-repositório)
- [Fundamentação técnica](#fundamentação-técnica)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura de diretórios esperada](#estrutura-de-diretórios-esperada)
- [Como executar](#como-executar)
- [Parâmetros de ajuste](#parâmetros-de-ajuste)
- [Limitações e problemas conhecidos](#limitações-e-problemas-conhecidos)
- [Próximos passos](#próximos-passos)
- [Referências](#referências)

---

## Motivação e escopo

O problema atacado é: **quantas pessoas distintas passaram por um ambiente, e quem eram elas?**

Contar pessoas por detecção simples (quantas caixas aparecem no frame) responde apenas "quantas estão agora". Para responder "quantas passaram no total" é preciso **identidade persistente** — reconhecer que a pessoa que apareceu no frame 10 é a mesma do frame 300. O repositório explora duas famílias de solução para isso:

| Abordagem | Identidade baseada em | Persiste entre sessões? | Custo computacional |
|---|---|---|---|
| **Reconhecimento facial** (`face_recognition`/dlib) | *embedding* de 128 dimensões do rosto | Sim — o rosto é salvo em disco | Alto (CPU) |
| **Detecção + rastreio** (YOLO11 + tracker) | ID de tracklet atribuído pelo rastreador | Não — o ID se perde quando a pessoa sai do quadro | Médio (acelera em GPU) |

As duas são complementares: a facial dá identidade real mas exige o rosto visível e frontal; o YOLO detecta o corpo inteiro (funciona de costas, longe, com oclusão parcial) mas não sabe *quem* é.

> **Escopo:** este é um repositório de **protótipos de pesquisa**, não um sistema pronto para produção. Veja [Limitações](#limitações-e-problemas-conhecidos).

---

## Visão geral da arquitetura

### Pipeline de reconhecimento facial

```
frame (BGR)
   │
   ├─► redimensiona para 25%          ← reduz custo do detector em ~16x
   │
   ├─► converte BGR → RGB             ← dlib espera RGB
   │
   ├─► face_locations()               ← detector HOG: onde estão os rostos
   │
   ├─► face_encodings()               ← rede neural: rosto → vetor de 128 dims
   │
   ├─► face_distance() vs. base       ← distância euclidiana contra os conhecidos
   │      │
   │      ├─ menor distância < 0.6 ──► pessoa CONHECIDA (usa o nome do arquivo)
   │      └─ senão ─────────────────► pessoa DESCONHECIDA
   │
   └─► desenha caixas + rótulos e exibe
```

### Pipeline de detecção e rastreio (YOLO)

```
frame → resize (1020x600) → YOLO11-s .track(classes=0) → caixas + track_id → desenha
                                        ▲
                                 classe 0 do COCO = "person"
```

---

## Scripts do repositório

| Arquivo | Fonte de vídeo | O que faz | Maturidade |
|---|---|---|---|
| `reconhecimentoPorImagem copy.py` | 2 imagens estáticas | Prova de conceito mínima: localiza um rosto, gera os *encodings* de duas fotos e imprime se são a mesma pessoa. | Didático |
| `reconhecimentoPorVideo.py` | arquivo `.mp4` | Processa 1 frame a cada 30, reconhece rostos e **salva os frames anotados** em `output_frames/`. Útil para análise offline/lote. | Experimental |
| `reconhecimentoPorWebcam.py` | arquivo `.mp4` (apesar do nome) | Versão em "tempo real" com processamento em frames alternados e sobreposição de rótulos. | Experimental / com bugs |
| `contagemPessoas.py` | arquivo `.mp4` | Detecção e rastreio de pessoas com **YOLO11-s**, exibindo `track_id` e classe. Contém o esqueleto (ainda inativo) para contagem por cruzamento de linha. | Experimental |
| `contagemPessoasDroidCam.py` | **stream HTTP** (DroidCam) | **Script principal.** Reconhecimento facial ao vivo, salvamento automático de desconhecidos, interface de nomenclatura interativa e contagem de pessoas únicas. | Mais completo |

### `contagemPessoasDroidCam.py` em detalhe

É o script que consolida o aprendizado dos demais. Funcionalidades:

1. **Carga da base** — lê todos os `.jpg/.png/.jpeg` de `faces/`, ignorando arquivos com prefixo `desconhecido_`. O **nome do arquivo vira o nome da pessoa** (`arthur.jpg` → `arthur`). Arquivos sem rosto detectável são avisados e ignorados, em vez de quebrar a execução.
2. **Aprendizado incremental de desconhecidos** — quando um rosto não bate com ninguém da base, o recorte é salvo como `faces/desconhecido_<timestamp>.jpg`. Antes de salvar, o *encoding* é comparado com os já salvos **na sessão atual**, evitando dezenas de arquivos da mesma pessoa.
3. **Interface de nomenclatura (tecla `N`)** — percorre os arquivos `desconhecido_*`, exibe cada um e pergunta no terminal o nome a atribuir, com opções de pular (`p`) ou deletar (`d`). Ao final, recarrega a base automaticamente.
4. **Contagem de pessoas únicas** — mantém um `set` com todos os nomes conhecidos já vistos e soma a quantidade de desconhecidos distintos salvos:

   ```
   total = |conhecidos vistos na sessão| + |desconhecidos únicos salvos|
   ```

5. **Otimização** — processa apenas frames alternados (`process_current_frame`) e roda a detecção em uma cópia reduzida a 25%, multiplicando as coordenadas por 4 na hora de desenhar.

---

## Fundamentação técnica

### Por que *embeddings* e não comparação de pixels

A biblioteca [`face_recognition`](https://github.com/ageitgey/face_recognition) encapsula o **dlib**. O reconhecimento acontece em duas redes distintas:

- **Detecção** (`face_locations`) — por padrão usa **HOG** (*Histogram of Oriented Gradients*) + SVM linear: rápido em CPU, mas sensível a rostos de perfil e baixa iluminação. Existe também o modo `model="cnn"`, muito mais robusto e muito mais lento sem GPU.
- **Codificação** (`face_encodings`) — uma ResNet treinada com *triplet loss* projeta o rosto alinhado em um vetor de **128 dimensões**. O treino otimiza para que rostos da mesma pessoa fiquem próximos nesse espaço e de pessoas diferentes fiquem distantes, independentemente de iluminação, ângulo e expressão.

A comparação, então, é uma simples **distância euclidiana** entre vetores:

```
d = ||encoding_A − encoding_B||₂
```

### O limiar de 0.6

O valor `face_match_threshold = 0.6` usado em todos os scripts é o padrão recomendado pelo dlib, calibrado no benchmark **LFW** (*Labeled Faces in the Wild*), onde alcança ~99,38% de acurácia. A interpretação prática:

| Distância | Interpretação |
|---|---|
| `< 0.4` | Correspondência muito forte |
| `0.4 – 0.6` | Correspondência aceita (padrão) |
| `> 0.6` | Pessoas diferentes |

Abaixar o limiar (ex.: `0.5`) reduz **falsos positivos** (confundir duas pessoas) ao custo de mais **falsos negativos** (não reconhecer alguém conhecido). Subir faz o inverso. Em bases pequenas e ambientes controlados, `0.5` costuma ser mais seguro.

### A função `face_confidence`

A distância euclidiana não é uma probabilidade. A função `face_confidence` converte a distância em um percentual **apenas para exibição ao usuário** — é uma heurística, não uma medida calibrada de confiança:

```python
range_val  = 1.0 - threshold              # 0.4
linear_val = (1.0 - distance) / (range_val * 2.0)

if distance > threshold:
    return linear_val * 100               # região de "não match": queda linear
else:
    # dentro do match, aplica uma curva que empurra os valores para perto de 100%
    return (linear_val + (1.0 - linear_val) * ((linear_val - 0.5) * 2) ** 0.2) * 100
```

Ou seja: fora do limiar o decaimento é linear; dentro do limiar aplica-se uma correção não linear que satura o valor perto de 100%. **Não trate esse número como probabilidade estatística.**

### YOLO11 e rastreio

`contagemPessoas.py` usa `model.track(frame, persist=True, classes=0)`:

- **`yolo11s.pt`** — variante *small* do YOLO11, treinada no **COCO** (80 classes). Bom equilíbrio entre velocidade e precisão para CPU/GPU modesta.
- **`classes=0`** — restringe a inferência à classe `person` do COCO, descartando as outras 79.
- **`persist=True`** — informa ao rastreador (BoT-SORT por padrão) que os frames são de uma sequência contínua, permitindo propagar `track_id` entre frames. Sem isso, cada frame seria tratado como imagem isolada e os IDs seriam reatribuídos do zero.
- **`count % 3 != 0: continue`** — processa 1 de cada 3 frames para ganhar velocidade. É um trade-off: pula frames que o rastreador poderia usar para manter a associação, então movimentos rápidos aumentam a troca de ID (*ID switch*).

---

## Requisitos

- **Python 3.8 – 3.11** (o `dlib` costuma falhar em versões mais novas sem wheels pré-compiladas)
- **CMake** e um compilador C++ — necessários para compilar o `dlib`
  - Windows: *Visual Studio Build Tools* com o workload "Desenvolvimento para desktop com C++"
  - Linux: `sudo apt install build-essential cmake libopenblas-dev liblapack-dev`
- Uma webcam, um arquivo de vídeo ou o app **DroidCam** no celular
- (Opcional) GPU NVIDIA com CUDA — acelera bastante o YOLO

### Dependências Python

| Pacote | Usado em |
|---|---|
| `face_recognition` (+ `dlib`) | todos os scripts `reconhecimento*` e `contagemPessoasDroidCam.py` |
| `opencv-python` | todos |
| `numpy` | todos |
| `ultralytics` | `contagemPessoas.py` |
| `cvzone` | `contagemPessoas.py` |

---

## Instalação

```bash
git clone https://github.com/arthruur/reconhecimento-facial-contagem-pessoas.git
cd reconhecimento-facial-contagem-pessoas

# ambiente virtual (recomendado)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-python numpy cmake dlib face_recognition ultralytics cvzone
```

> **Windows:** se `pip install dlib` falhar na compilação, instale primeiro o *Visual Studio Build Tools* ou use uma wheel pré-compilada compatível com a sua versão do Python.

O peso `yolo11s.pt` (~19 MB) já está versionado no repositório — não é necessário baixá-lo.

---

## Estrutura de diretórios esperada

Os scripts assumem convenções que **não estão versionadas** e precisam ser criadas localmente:

```
reconhecimento-facial-contagem-pessoas/
├── faces/                     ← criado automaticamente pelo script DroidCam
│   ├── arthur.jpg             ← 1 rosto por arquivo; nome do arquivo = nome da pessoa
│   ├── kauan.jpg
│   └── desconhecido_<ts>.jpg  ← gerado em runtime, aguardando nomeação
├── output_frames/             ← criado por reconhecimentoPorVideo.py
├── seu_video.mp4              ← forneça o seu; os vídeos não estão no repositório
└── yolo11s.pt                 ← versionado
```

**Boas práticas para as imagens em `faces/`:**

- exatamente **um rosto por imagem** (o código usa `face_encodings(...)[0]`);
- rosto frontal, bem iluminado, sem óculos escuros ou máscara;
- resolução suficiente para o rosto ocupar boa parte do quadro;
- nome do arquivo sem espaços e sem o prefixo reservado `desconhecido_`.

---

## Como executar

### 1. Comparar duas imagens (prova de conceito)

Coloque duas fotos em `faces/` e ajuste os nomes no topo do arquivo:

```bash
python "reconhecimentoPorImagem copy.py"
```

Imprime as coordenadas do rosto detectado e `[True]`/`[False]` para a comparação.

### 2. Processar um arquivo de vídeo em lote

Edite `videoPath` em `reconhecimentoPorVideo.py` e rode:

```bash
python reconhecimentoPorVideo.py
```

Os frames anotados são gravados em `output_frames/`. Pressione `q` para encerrar.

### 3. Detecção e rastreio com YOLO

Edite o caminho em `cv2.VideoCapture(...)` dentro de `contagemPessoas.py` e rode:

```bash
python contagemPessoas.py
```

A janela mostra as caixas com classe e `track_id`. Mover o mouse sobre a janela imprime as coordenadas no terminal — útil para calibrar as linhas de contagem (`cy1`, `cy2`).

### 4. Reconhecimento ao vivo com DroidCam (principal)

1. Instale o **DroidCam** no celular e conecte-o à **mesma rede Wi-Fi** do computador.
2. Abra o app e anote o IP exibido.
3. Edite a linha 127 de `contagemPessoasDroidCam.py`:

   ```python
   droidcam_url = "http://SEU_IP:4747/video"
   ```

4. Execute:

   ```bash
   python contagemPessoasDroidCam.py
   ```

**Controles durante a execução:**

| Tecla | Ação |
|---|---|
| `N` | Abre a interface de nomenclatura dos rostos desconhecidos salvos |
| `Q` | Encerra a aplicação |

Na interface de nomenclatura, para cada rosto: digite um **nome** para salvá-lo na base, `p` para pular ou `d` para deletar.

---

## Parâmetros de ajuste

| Parâmetro | Onde | Padrão | Efeito |
|---|---|---|---|
| `face_match_threshold` | `face_confidence()` | `0.6` | Menor = mais rigoroso, menos falsos positivos |
| Limiar de dedupe de desconhecidos | `autosave_unknown_faces()` | `0.6` | Menor = salva mais variações da mesma pessoa |
| Fator de redimensionamento | `cv2.resize(..., fx=0.25, fy=0.25)` | `0.25` | Maior = mais preciso e mais lento (lembre de ajustar o `*= 4`) |
| `process_current_frame` | loop principal | alterna | Processa 1 frame a cada 2 |
| `count % 3` | `contagemPessoas.py` | `3` | Processa 1 frame a cada 3 no pipeline YOLO |
| `classes` | `model.track()` | `0` | Classes do COCO a detectar (`0` = pessoa) |
| Resolução de exibição | `cv2.resize(frame, (1020, 600))` | `1020x600` | Referência para as coordenadas das linhas de contagem |

---

## Limitações e problemas conhecidos

Registrados aqui de forma explícita para orientar quem for continuar o trabalho.

### Arquitetura e reprodutibilidade

- **Não há `requirements.txt`** — as versões das dependências não estão fixadas, o que compromete a reprodutibilidade (especialmente `dlib` e `ultralytics`).
- **Caminhos e IPs estão hardcoded** — o IP da DroidCam, os caminhos de vídeo e os nomes de arquivo estão no código-fonte. Migrar para argumentos de CLI ou variáveis de ambiente é o próximo ganho óbvio de usabilidade.
- **Os vídeos de teste não estão versionados** — os scripts referenciam arquivos (`WIN_20241025_*.mp4`, `WhatsApp Video 2025-09-24*.mp4`) ausentes do repositório; é necessário fornecer os seus.
- **Há código duplicado entre os scripts** — a classe `FaceRecognition` aparece em três variantes divergentes, sem base comum.

### Bugs identificados no código atual

- `reconhecimentoPorWebcam.py` — a chamada `math.pow(linearVal - 0.5) * 2, 0.2)` está com os parênteses trocados e levanta `TypeError` quando um rosto é reconhecido; além disso, `videoCapture.release()` está **dentro** do `while`, encerrando a captura na primeira iteração. O nome sugere webcam, mas a fonte é um arquivo de vídeo.
- `reconhecimentoPorVideo.py` — passa o frame em **BGR** direto para `face_recognition`, sem converter para RGB, o que degrada a detecção; `np.argmin` é chamado sem verificar se a base de rostos está vazia; a `face_confidence` definida no arquivo nunca é usada.
- `contagemPessoasDroidCam.py` — `recognized_people_in_frame` só é atribuída nos frames processados, então nos frames pulados o valor do frame anterior é reaproveitado; a contagem de desconhecidos é reiniciada a cada sessão (`saved_unknown_encodings` é limpa em `encode_faces`).
- `contagemPessoas.py` — as variáveis de contagem por linha (`cy1`, `cy2`, `offset`, `inp`, `enter`, `exp`, `exitp`) estão declaradas mas **não utilizadas**; as linhas de referência estão comentadas. A contagem de entradas/saídas ainda não foi implementada.

### Limitações intrínsecas da abordagem

- O detector **HOG** falha com rostos de perfil, muito pequenos, mal iluminados ou parcialmente ocluídos.
- O reconhecimento roda em **CPU**; o custo cresce linearmente com o número de rostos cadastrados e é o gargalo do sistema.
- A contagem de "pessoas únicas" só é confiável se **cada pessoa mostrar o rosto** para a câmera pelo menos uma vez.
- O salvamento automático de desconhecidos pode gerar duplicatas quando a mesma pessoa aparece em ângulos muito diferentes (a distância entre os *encodings* ultrapassa o limiar de dedupe).

### Privacidade e uso ético

Este projeto processa e **armazena em disco dados biométricos** (imagens de rostos). No Brasil, dados biométricos são classificados como **dados pessoais sensíveis** pela **LGPD (Lei 13.709/2018, art. 5º, II)**. Qualquer uso fora de um ambiente estritamente experimental e controlado exige base legal adequada, consentimento explícito dos titulares e política definida de retenção e descarte. **Não utilize em ambientes públicos ou com terceiros sem autorização.**

---

## Próximos passos

- [ ] Adicionar `requirements.txt` com versões fixadas e um `.gitignore` (excluindo `faces/`, `output_frames/`, `*.mp4`)
- [ ] Corrigir os bugs listados acima
- [ ] Unificar a classe `FaceRecognition` em um módulo único reutilizável
- [ ] Parametrizar fonte de vídeo, IP e limiares via `argparse` ou arquivo de configuração
- [ ] Implementar a contagem de entrada/saída por cruzamento de linha em `contagemPessoas.py`
- [ ] **Fundir os dois pipelines**: usar o YOLO para detectar e rastrear corpos e o reconhecimento facial para atribuir identidade ao tracklet — resolvendo as fraquezas de ambos
- [ ] Persistir os *encodings* em cache (`.pkl`/`.npy`) para evitar reprocessar a base a cada execução
- [ ] Registrar os eventos (pessoa, timestamp) em CSV ou banco de dados
- [ ] Avaliar quantitativamente o sistema (precisão, revocação, matriz de confusão) em um vídeo anotado

---

## Referências

- [`face_recognition` — Adam Geitgey](https://github.com/ageitgey/face_recognition)
- [dlib — Deep Face Recognition](http://dlib.net/face_recognition.py.html)
- Schroff, Kalenichenko & Philbin (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR — base teórica do *triplet loss* usado nos *embeddings*.
- Dalal & Triggs (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR — base do detector HOG.
- [Ultralytics YOLO11 — documentação](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics — Object Tracking](https://docs.ultralytics.com/modes/track/)
- [Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
