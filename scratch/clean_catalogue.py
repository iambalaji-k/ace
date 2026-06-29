import re

def clean_catalogue():
    filepath = "D:/Vibe Coding/ace/Ace_Heuristic_Catalogue_v1.0.md"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Define heuristics to remove (by ID)
    to_remove = {
        "H101", "H116", "H118",
        "H201", "H207", "H212",
        "H302", "H309", "H312", "H318",
        "H404", "H407", "H409", "H414",
        "H509", "H709"
    }

    # Split the file by ### to process each section/heuristic
    sections = re.split(r'\n(### H\d+:[^\n]*)', content)
    
    # The first section is the header/intro
    new_sections = [sections[0]]

    # Iterate over the split sections
    for i in range(1, len(sections), 2):
        header = sections[i]
        body = sections[i+1]
        
        # Extract heuristic ID (e.g., H101)
        h_id_match = re.search(r'### (H\d+):', header)
        if h_id_match:
            h_id = h_id_match.group(1)
            if h_id in to_remove:
                # Skip/remove this heuristic
                continue
            
            # Special processing for H117
            if h_id == "H117":
                header = header.strip() + " - unless they have discarded the ace already."
            
            # Special processing for H223
            if h_id == "H223":
                body = """
* **Purpose**: After collecting cards from an interrupted trick (our hand just grew), select the safest lead by choosing the suit for which non interruption is highly likely.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We just collected cards and must now lead.
* **Evaluation Formula**:
  * Lead lowest card of suit $S$ that maximizes the expected follow probability of all other active players: $+150.0$
  * If any active player is confirmed void in suit $S$: $-200.0$
* **Weight**: $150.0$
* **Rationale**: After collecting cards, our hand is bloated. We must lead a suit that all active players can follow, ensuring the trick is discarded and we do not collect any more cards.
"""

        # Append cleaned header and body
        new_sections.append("\n" + header + body)

    # Rejoin the content
    new_content = "".join(new_sections)

    # Remove all trump-related terminology
    # Replace Spades trump conservation rationale or clean up mentions of "trump"
    # "trump suit", "trump", "trumping", etc.
    new_content = re.sub(r'(?i)\btrump(ing|s)?\b', 'interruption', new_content)
    new_content = re.sub(r'trump-like', 'interruption-like', new_content)
    new_content = re.sub(r'undertrumping', 'underplaying', new_content)
    new_content = re.sub(r'Spade conservation \(trump\)', 'Spade conservation', new_content)
    
    # Clean up the count in the header
    # Let's count how many ### H\d+: lines are left
    remaining_count = len(re.findall(r'### H\d+:', new_content))
    total_count_str = f"**Total entries: {remaining_count + 1}** (1 Core Utility + {remaining_count} Heuristics)"
    
    # Replace the old total count in the header
    new_content = re.sub(r'\*\*Total entries:.*', total_count_str, new_content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Cleaned catalogue successfully. {remaining_count} heuristics remaining.")

if __name__ == "__main__":
    clean_catalogue()
