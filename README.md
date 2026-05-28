# Novel Character Memory

Scrape RoyalRoad chapters, summarize each chapter locally, and build character
memory that can be queried up to any chapter.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For summarization, provide a Hugging Face GGUF model repo supported by
`llama.cpp`. A small instruct model quantized to Q4 is a practical starting
point for a 4 GB GPU.

## Usage

Scrape a novel:

```powershell
python -m novel_memory scrape --title "Practical Guide To Evil" --start-url "https://www.royalroad.com/fiction/..." --max-chapters 100
```

Summarize chapters and update character memory:

```powershell
python -m novel_memory summarize --novel practical_guide_to_evil --model-repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF --model-file "*Q4_K_M.gguf" --gpu-layers 20
```

Convert older `chapters_1_20.json` batch files into the new per-chapter layout:

```powershell
python -m novel_memory migrate-batches --novel practical_guide_to_evil --title "Practical Guide To Evil"
```

Look up what is known about a character by a specific chapter:

```powershell
python -m novel_memory character --novel practical_guide_to_evil --chapter 40 --name "Arn"
```

All generated project data lives under:

```text
novels_extracted/<novel_slug>/
```

## Output Layout

```text
novels_extracted/<novel_slug>/
  metadata.json
  chapters/
    chapter_0001.json
  summaries/
    chapter_0001.json
  characters/
    arn.json
  indexes/
    characters.json
```

Raw chapter text is kept separate from summaries so the summaries can be
regenerated with a better model later.


from now on teh commits arein this form:
feat: = new validation behavior
refactor: = same behavior, cleaner structure
fix: = bug fix
test: = adding or updating tests
docs: chnage in docs
