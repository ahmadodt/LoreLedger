# Lore Ledger — Next Steps Plan

## Main Project Title

**Can AI maintain a continuously evolving understanding of a long fictional world over thousands of chapters?**

## Core Direction

Build a system that can scrape, summarize, store, retrieve, and reason over long fictional stories.
The long-term goal is to create a conversational AI agent that understands the evolving world, characters, events, and lore of a story across many chapters.

---

# Priority Roadmap

## 1. Conversational Story Agent

This is the main priority.

The goal is to let the user ask questions about a story and get answers based on the saved chapters, summaries, character information, and lore.

Example questions:

* What happened to this character?
* Why does character X hate character Y?
* When was this item introduced?
* What is the current political situation?
* What does this character currently know?
* Summarize the latest arc.

### Next steps

* Store chapter summaries in a searchable format.
* Create embeddings for chapter summaries and important chunks.
* Build a simple RAG pipeline.
* Add citations or references to chapter numbers.
* Add a chat interface.
* Make the model answer only using retrieved story information.

---

## 2. Agentic Pipeline

Build the project as a modular AI pipeline.

The system should have separate components for:

* scraping chapters
* cleaning text
* chunking chapters
* summarizing chapters
* extracting characters
* updating character memory
* embedding story data
* retrieving relevant information
* generating answers

### Next steps

* Define each module clearly.
* Save intermediate outputs as JSON.
* Make each step reusable.
* Add logging and debugging.
* Make it easy to rerun only one part of the pipeline.

---

## 3. Character Psychological Profiles

Track how characters change over time.

This can include:

* goals
* fears
* motivations
* personality traits
* loyalties
* relationships
* emotional state
* major turning points

### Next steps

* Start with simple character profiles.
* Update profiles after each chapter or group of chapters.
* Store “current state” and “history.”
* Track how motivations and relationships change.
* Later, compare old and new profiles to detect character development.

---

## 4. Incremental Story State Updating

Instead of regenerating everything from scratch, update the story memory step by step.

The system should read new chapters and update only the relevant parts of the memory.

### Next steps

* Keep a current story state file.
* After each chapter batch, extract new facts.
* Merge new facts into existing memory.
* Avoid overwriting useful old information.
* Mark outdated information when newer events change it.

---

## 5. Entity Graph

Build a graph of the story world.

Possible nodes:

* characters
* factions
* locations
* items
* powers
* events

Possible edges:

* belongs to
* enemy of
* ally of
* located in
* owns
* killed by
* trained by
* related to

### Next steps

* Start with character-to-character relationships.
* Add locations and factions later.
* Store graph data in JSON first.
* Later use a graph database or visualization tool.
* Use the graph for better retrieval and reasoning.

---

## 6. Narrative Memory System

Create layered memory for the story.

Possible memory layers:

* recent chapter memory
* arc memory
* character memory
* world/lore memory
* long-term story memory

### Next steps

* Separate recent information from long-term information.
* Store summaries at different levels.
* Create arc-level summaries.
* Retrieve from different memory layers depending on the question.
* Use this later to improve the conversational agent.

---

## 7. Fantasy-Aware NLP

Fantasy stories have difficult names, titles, aliases, races, powers, and places.

The system should handle things like:

* “The Black Swordsman”
* “the young prince”
* “Arthur”
* “the heir of flame”

possibly referring to the same character.

### Next steps

* Improve entity extraction.
* Add alias detection.
* Track titles and nicknames.
* Detect fantasy-specific entities like powers, artifacts, clans, kingdoms, and monsters.
* Compare simple NLP models with LLM-based extraction.

---

## 8. Wiki Generation

Generate automatic wiki-style pages from the story memory.

Possible pages:

* character pages
* location pages
* faction pages
* item pages
* power system pages
* arc summaries

### Next steps

* Generate one character page from stored memory.
* Add chapter references.
* Keep pages updated as new chapters are processed.
* Add “current status” and “history” sections.
* Later, make a small local web UI.

---

## 9. Multi-Story Cross-Analysis

Later, compare multiple stories.

Possible comparisons:

* pacing
* character growth
* trope usage
* theme frequency
* power scaling
* dialogue density
* arc structure

### Next steps

* First make the system work for one story.
* Then support multiple story folders.
* Store separate databases per story.
* Add high-level story comparison features later.

---

## 10. Arc Detection

Detect story arcs automatically.

Examples:

* training arc
* tournament arc
* revenge arc
* war arc
* political arc
* romance arc

### Next steps

* Group chapters into arcs manually first.
* Generate arc summaries.
* Later try automatic arc boundary detection.
* Use changes in characters, locations, goals, and conflicts to detect arc shifts.

---

## 11. Theme and Symbol Tracking

Track recurring ideas in the story.

Examples:

* revenge
* sacrifice
* corruption
* loyalty
* freedom
* fate
* recurring symbols or objects

### Next steps

* Start with manual theme labels.
* Later use LLM extraction.
* Track which chapters mention each theme.
* Summarize how a theme develops over time.

---

# Removed / Not Planned For Now

These ideas are not part of the current plan:

* Timeline consistency engine
* Trust / confidence scoring
* Prediction system

They can be reconsidered later, but they are not priorities now.

---

# Suggested MVP

## MVP 1: Conversational RAG Story Agent

The first useful version should include:

* scraped chapters
* chapter text stored in JSON
* chapter summaries
* embeddings for summaries/chunks
* basic retrieval
* chat-based question answering
* chapter references in answers

This MVP should answer questions like:

* What happened in the latest chapters?
* Who is character X?
* What is the relationship between X and Y?
* Where did this event happen?
* What is currently happening in the story?

---

# Immediate Next Steps

1. Finalize the JSON structure for storing stories and chapters.
2. Store chapter text and chapter summaries.
3. Add embeddings for chapter summaries.
4. Build a simple retrieval function.
5. Create a basic chat function that answers using retrieved context.
6. Add chapter references to answers.
7. Improve character summaries.
8. Start storing character profiles separately.
9. Add incremental updates later.
10. Expand into entity graphs and narrative memory once the basic agent works.

---

# Project Identity

This project is not just a normal summarizer.

It is an evolving story-memory system.

The main challenge is:

**How can an AI keep an accurate, useful, and continuously updated understanding of a fictional world as the story grows over hundreds or thousands of chapters?**




*** a good idea to add, is an agent that could cathc poblems inside stories, example in the power system or in the story or in the progression or something missing, later we can delve deepr in this after i understand and refine the RAG