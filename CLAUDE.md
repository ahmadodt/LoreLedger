# LoreLedger — Claude Guidelines

## What is LoreLedger
LoreLedger is a local-first story memory system for long web novels (primarily RoyalRoad).
It builds and maintains a structured understanding of a fictional world across hundreds of 
chapters using a local GGUF model — no external APIs.

## Pipeline
Scrape → Summarize → Index → Ask

1. scraper.py — crawls RoyalRoad chapter by chapter, saves raw text as JSON
2. summarizer.py — runs chapters through a local GGUF model, produces structured JSON 
   summaries with typed events, character updates, locations, and continuity flags
3. rag.py — builds a TF-IDF searchable index from summaries and character timelines
4. streamlit_app.py — web UI with tabs for Scrape, Novel, Character Memory, Ask

## Key Conventions
- All data is stored as plain JSON files — no database
- Novel data lives under novels_extracted/<novel-slug>/
- Every extracted event must include evidence quoted exactly from the chapter text
- chapter_summary is always a dict with 5 fields: situation, conflict, turning_point, 
  consequence, hook — never a plain string
- No backward compatibility code — old