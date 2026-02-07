import re

def greedy_align_and_filter(rel_words, salient_tokens):
    """
    Greedy sequential matcher that:
      • merges consecutive rel_words to match a salient token
      • skips irrelevant rel_words
      • ignores case and punctuation
    """
    def normalize(s):
        # Keep alphanumeric characters for matching
        # return re.sub(r'[^a-z0-9]', '', s.lower())
        return s.lower()

    rel_idx = 0
    salient_idx = 0
    result = []
    
    while rel_idx < len(rel_words) and salient_idx < len(salient_tokens):
        target = normalize(salient_tokens[salient_idx])
        if not target:
            salient_idx += 1
            continue

        found_match = False
        # Iterate through possible start points in the remainder of rel_words
        for i in range(rel_idx, len(rel_words)):
            
            # --- Attempt to build a match starting from rel_words[i] ---
            merged_normalized = ""
            collected_words = []
            
            for j in range(i, len(rel_words)):
                word = rel_words[j]
                normalized_word = normalize(word)

                # If the current normalized word is empty, we can't start a match with it.
                # However, we can include it if we've already started building a match.
                if not normalized_word:
                    if merged_normalized:
                        collected_words.append(word)
                    continue

                # Check if adding the next normalized word would make the sequence invalid.
                if not target.startswith(merged_normalized + normalized_word):
                    # This sequence starting from `i` doesn't work. Break from this inner loop.
                    break

                # The sequence is still valid, so add the word and its normalized form.
                merged_normalized += normalized_word
                collected_words.append(word)

                # Check if we have a complete match.
                if merged_normalized == target:
                    result.extend(collected_words)
                    rel_idx = j + 1
                    salient_idx += 1
                    found_match = True
                    break # Match found, break from the inner `j` loop.
            
            if found_match:
                # The `salient_token` was matched. Break from the `i` loop to get the next `salient_token`.
                break
        
        if not found_match:
            # We scanned from rel_idx to the end and couldn't find a match for the current salient_token.
            # This means it's not in the rel_words, so we skip it and move to the next one.
            salient_idx += 1

    return result

if __name__ == "__main__":
    # Example usage:
    rel_words = ['Wh', 'at', 'the', 'lowest', 'number', 'yard', 'line', 'that', 'you', 'can', 'see', '?', '\n', 'Reference', 'O', 'CR', 'token', ':', '', '\n', 'Answer', 'the', 'question', 'using', 'a', 'single', 'word', 'or', 'phrase', '.']
    salient_tokens_filtered = ['what', 'number', 'yard', 'line', 'see', '?', 'reference', 'ocr', 'token', ':', 'answer', 'question', 'using', 'word', 'phrase']

    output = greedy_align_and_filter(rel_words, salient_tokens_filtered)
    
    expected_output = ['Wh', 'at', 'number', 'yard', 'line', 'see', '?', 'Reference', 'O', 'CR', 'token', ':', 'Answer', 'question', 'using', 'word', 'phrase']
    
    print(f"Got:      {output}")
    print(f"Expected: {expected_output}")
    print(f"Match:    {output == expected_output}")
    
# Many-to-many matching (instead of many-to-one style given above)

# def greedy_align_and_filter(rel_words, salient_tokens):
#     """
#     Greedy sequential matcher that handles many-to-many merges.
#     """
#     def normalize(s):
#         return re.sub(r'[^a-z0-9]', '', s.lower())

#     rel_idx = 0
#     salient_idx = 0
#     result = []

#     while rel_idx < len(rel_words) and salient_idx < len(salient_tokens):
#         # Try to match one or more rel_words to one or more salient_tokens
        
#         # Find the longest possible match starting from the current indices
#         best_rel_count = 0
#         best_salient_count = 0
        
#         # Greedily extend salient_tokens
#         salient_merge = ""
#         for s_count in range(1, len(salient_tokens) - salient_idx + 1):
#             salient_merge = "".join(normalize(s) for s in salient_tokens[salient_idx : salient_idx + s_count])
            
#             # Greedily extend rel_words to match the merged salient_tokens
#             rel_merge = ""
#             for r_count in range(1, len(rel_words) - rel_idx + 1):
#                 # Skip over words in rel_words that are irrelevant
#                 temp_rel_idx = rel_idx
#                 merged_rel_words = []
                
#                 # This inner part is complex: find a subsequence in rel_words that matches
#                 # For simplicity, this example assumes a direct merge, but a real solution
#                 # would need the inner loops from the previous version here.
                
#                 rel_merge = "".join(normalize(r) for r in rel_words[rel_idx : rel_idx + r_count])

#                 if rel_merge == salient_merge:
#                     # Found a potential match, see if it's the longest so far
#                     best_rel_count = r_count
#                     best_salient_count = s_count
#                     # In a real greedy implementation, you might stop at the first match.
#                     # For "longest", you'd continue searching.
        
#         if best_rel_count > 0:
#             # Add the matched words from rel_words to the result
#             matched_words = rel_words[rel_idx : rel_idx + best_rel_count]
#             result.extend(matched_words)
#             rel_idx += best_rel_count
#             salient_idx += best_salient_count
#         else:
#             # No match found, advance one of the pointers to avoid an infinite loop.
#             # This is a simplification; deciding which to advance is non-trivial.
#             salient_idx += 1

#     return result

# if __name__ == "__main__":
#     # This simplified example will not work correctly with the above conceptual code.
#     # The previous version of the code is better suited for the one-to-many case.
#     rel_words = ['Wh', 'at', 'the', 'lowest', 'number', 'yard']
#     salient_tokens_filtered = ['what', 'number', 'ya','rd']
    
#     print("The current algorithm does not support merging salient tokens.")
#     print("It would incorrectly produce: ['Wh', 'at', 'number']")
#     print("Handling this case requires a many-to-many alignment algorithm or pre-processing the salient tokens.")
