import streamlit as st
import json
import spacy
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from json import *
import io
nltk.download('punkt')

# Load spaCy model for SVO extraction
nlp = spacy.load("en_core_web_sm")

# Function to extract SVO relations
import streamlit as st
import spacy
import json
import pandas as pd
from io import StringIO


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def compute_confidence(subject, verb, obj):
    confidence = 1.0
    if subject.dep_ in {"nsubj", "nsubjpass"}:
        confidence += 0.5
    if verb.pos_ == "VERB":
        confidence += 0.5
    if obj.dep_ in {"dobj", "attr", "pobj"}:
        confidence += 0.5
    if subject.dep_ not in {"nsubj", "nsubjpass"}:
        confidence -= 0.5
    if verb.pos_ != "VERB":
        confidence -= 0.5
    if obj.dep_ not in {"dobj", "attr", "pobj"}:
        confidence -= 0.5
    if subject.head == verb:
        confidence += 0.2
    if obj.head == verb:
        confidence += 0.2
    distance = abs(subject.i - verb.i) + abs(verb.i - obj.i)
    confidence -= (distance * 0.05)
    if subject.ent_type_:
        confidence += 0.1
    if obj.ent_type_:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence / 2.5))
    return confidence

def extract_relations_with_confidence(text, confidence_threshold=0.4):
    doc = nlp(text)
    relations_with_confidence = []

    for sent in doc.sents:
        subjects = []
        verbs = []
        objects = []
        preps = []
        adjectives = []
        compounds = []
        possessives = []
        appositives = []
        conjuncts = []
        aux_verbs = []
        adverbs = []

        for token in sent:
            if "subj" in token.dep_:
                subjects.append(token)
            if token.pos_ == "VERB":
                verbs.append(token)
            if "obj" in token.dep_ or "attr" in token.dep_:
                objects.append(token)
            if token.dep_ == "prep":
                preps.append(token)
            if token.dep_ == "amod":
                adjectives.append(token)
            if token.dep_ == "compound":
                compounds.append(token)
            if token.dep_ == "poss":
                possessives.append(token)
            if token.dep_ == "appos":
                appositives.append(token)
            if token.dep_ in {"cc", "conj"}:
                conjuncts.append(token)
            if token.dep_ == "aux":
                aux_verbs.append(token)
            if token.dep_ == "advmod":
                adverbs.append(token)

        # Filter and validate relations
        for verb in verbs:
            for subject in subjects:
                for obj in objects:
                    # Skip if subject, verb, and object are too similar
                    if subject.text == obj.text or subject.text == verb.text or verb.text == obj.text:
                        continue

                    confidence = compute_confidence(subject, verb, obj)
                    if confidence >= confidence_threshold:
                        relations_with_confidence.append({
                            "subject": subject.text,
                            "verb": verb.text,
                            "object": obj.text,
                            "confidence": confidence
                        })

        for prep in preps:
            pobj = [child for child in prep.children if child.dep_ == "pobj"]
            if pobj:
                if prep.head.text == pobj[0].text or prep.text == pobj[0].text:
                    continue
                
                confidence = compute_confidence(prep.head, prep, pobj[0])
                if confidence >= confidence_threshold:
                    relations_with_confidence.append({
                        "subject": prep.head.text,
                        "verb": prep.text,
                        "object": pobj[0].text,
                        "confidence": confidence
                    })

        for adj in adjectives:
            adj_mod = " ".join(
                [child.text for child in adj.head.children if child.dep_ in {"nummod", "compound"}]
            )
            if adj_mod:
                if adj.head.text == adj_mod or adj.text == adj_mod:
                    continue

                confidence = compute_confidence(adj.head, adj, nlp(adj_mod)[0])
                if confidence >= confidence_threshold:
                    relations_with_confidence.append({
                        "subject": adj.head.text,
                        "verb": adj.text,
                        "object": adj_mod,
                        "confidence": confidence
                    })
            else:
                if adj.head.text == adj.text:
                    continue
                
                confidence = compute_confidence(adj.head, adj, adj.head)
                if confidence >= confidence_threshold:
                    relations_with_confidence.append({
                        "subject": adj.head.text,
                        "verb": adj.text,
                        "object": adj.head.text,
                        "confidence": confidence
                    })

        for comp in compounds:
            if comp.head.pos_ == "NOUN":
                if comp.head.text == comp.text:
                    continue
                
                confidence = compute_confidence(comp.head, comp, comp.head)
                if confidence >= confidence_threshold:
                    relations_with_confidence.append({
                        "subject": comp.head.text,
                        "verb": comp.text,
                        "object": comp.head.text,
                        "confidence": confidence
                    })

        for poss in possessives:
            if poss.head.text == poss.text:
                continue
            
            confidence = compute_confidence(poss.head, poss, poss.head)
            if confidence >= confidence_threshold:
                relations_with_confidence.append({
                    "subject": poss.head.text,
                    "verb": "'s",
                    "object": poss.text,
                    "confidence": confidence
                })
                

        for appos in appositives:
            if appos.head.text == appos.text:
                continue
            
            confidence = compute_confidence(appos.head, appos, appos.head)
            if confidence >= confidence_threshold:
                relations_with_confidence.append({
                    "subject": appos.head.text,
                    "verb": ",",
                    "object": appos.text,
                    "confidence": confidence
                })

        for conj in conjuncts:
            if conj.dep_ == "conj" and conj.head.pos_ == "NOUN":
                if conj.head.text == conj.text:
                    continue
                
                confidence = compute_confidence(conj.head, conj, conj.head)
                if confidence >= confidence_threshold:
                    relations_with_confidence.append({
                        "subject": conj.head.text,
                        "verb": "and",
                        "object": conj.text,
                        "confidence": confidence
                    })

        for aux in aux_verbs:
            if aux.head.pos_ == "VERB":
                if aux.head.text == aux.text:
                    continue
                
                confidence = compute_confidence(aux.head, aux, aux.head)
                if confidence >= confidence_threshold:
                    relations_with_confidence.append({
                        "subject": aux.head.text,
                        "verb": aux.text,
                        "object": aux.head.text,
                        "confidence": confidence
                    })

        for adv in adverbs:
            if adv.head.text == adv.text:
                continue
            
            confidence = compute_confidence(adv.head, adv, adv.head)
            if confidence >= confidence_threshold:
                relations_with_confidence.append({
                    "subject": adv.head.text,
                    "verb": adv.text,
                    "object": adv.head.text,
                    "confidence": confidence
                })

    # Filter out relations without valid values and exclude unwanted verb types
    filtered_relations = [
        rel for rel in relations_with_confidence
        if all(rel.values()) and not (nlp(rel["verb"])[0].pos_ in {"AUX", "CCONJ", "DET", "PART", "SCONJ"})
    ]

    # Sort the filtered relations by confidence
    filtered_relations.sort(key=lambda x: x["confidence"], reverse=True)

    # st.write("=====================================")
    # st.write("Filtered Relations :",filtered_relations)
    # st.write("=====================================")
    # Return only the filtered relations as a list
    return filtered_relations

# Function to compute ROUGE scores
def compute_rouge(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores

def compute_rouge_average(rouge_scores):
  
    """
    Compute average ROUGE scores from a dictionary of precomputed ROUGE scores.

    Args:
        rouge_scores (dict): A dictionary containing 'rouge1', 'rouge2', and 'rougeL' score lists.

    Returns:
        dict: A dictionary with average ROUGE scores.
    """
    # Calculate average ROUGE-1
    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    
    # Calculate average ROUGE-2
    avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    
    # Calculate average ROUGE-L
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])

    return {
        'avg_rouge1': avg_rouge1,
        'avg_rouge2': avg_rouge2,
        'avg_rougeL': avg_rougeL
    }

import pandas as pd

# Define unwanted symbols to filter out
unwanted_symbols = {'.', ',', '(', ')', '{', '}', '[', ']', '%','$', ';', ':','&',' ','"','\"'}

# Function to filter out unwanted symbols from a set of tuples
def filter_unwanted_symbols(tuples_set):
    return {t for t in tuples_set if not any(sym in t for sym in unwanted_symbols)}

from bert_score import score as bert_score
import streamlit as st

# def calculate_semantic_similarity(set1, set2):
#     st.write("====================")
#     st.write("Set 1: ", set1)
#     st.write("Set 2: ", set2)
#     st.write("====================")
    
#     unique_count = set(set1) | set(set2)  # Union of unique tuples from both sets
#     semantic_match_count = 0  # Count of semantically matched relations
#     matched_tups = set()  # To track already counted matches

#     # Iterate through each tuple in set2 to find matches in set1
#     for tup2 in set2:
#         subject2, verb2, object2 = tup2[:-1]  # Extract SVO from tup2 (excluding confidence)
        
#         for tup1 in set1:
#             if tup1 in matched_tups:
#                 continue
                
#             subject1, verb1, object1 = tup1[:-1]  # Extract SVO from tup1 (excluding confidence)

#             match_count = sum([subject1 == subject2, verb1 == verb2, object1 == object2])
#             if match_count > 0:
#                 matched_tups.add(tup1)  # Avoid double counting
#                 semantic_match_count += match_count / 3  # Partial overlap

#     total_unique_count = len(unique_count) + semantic_match_count
#     return total_unique_count

# def calculate_semantic_similarity_three_sets(set1, set2, set3):
#     st.write("====================")
#     st.write("Set 1: ", set1)
#     st.write("Set 2: ", set2)
#     st.write("Set 3: ", set3)
#     st.write("====================")
    
#     unique_count = set(set1) | set(set2) | set(set3)
#     semantic_match_count = 0
#     matched_tups = set()

#     for comparison_set in [set2, set3]:
#         for tup_comp in comparison_set:
#             subject_comp, verb_comp, object_comp = tup_comp[:-1]

#             match_count = 0
#             for tup1 in set1:
#                 if tup1 in matched_tups:
#                     continue
                
#                 subject1, verb1, object1 = tup1[:-1]
#                 match_count = sum([subject1 == subject_comp, verb1 == verb_comp, object1 == object_comp])
                
#                 if match_count > 0:
#                     matched_tups.add(tup1)
#                     semantic_match_count += match_count / 3  # Partial match weighting

#     total_unique_count = len(unique_count) + semantic_match_count
#     return total_unique_count


# def compute_hallucination_metrics(pred_words, ref_words, inp_words, input_relations, ref_relations, model_relations):
#     """
#     Computes hallucination metrics based on relation tuples, cosine similarity, and BERTScore.
#     """
#     # # Convert lists to sets of tuples and filter unwanted symbols
#     # pred_tuples = filter_unwanted_symbols(set(pred_words))
#     # ref_tuples = filter_unwanted_symbols(set(ref_words))
#     # inp_tuples = filter_unwanted_symbols(set(inp_words))
    
#     # Convert lists to sets of tuples and filter unwanted symbols
#     pred_tuples = filter_unwanted_symbols(set(model_relations))
#     ref_tuples = filter_unwanted_symbols(set(ref_relations))
#     inp_tuples = filter_unwanted_symbols(set(input_relations))
#     st.write("======CHM_METHOD=========")
#     st.write("Input Relations: ",input_relations)
#     st.write("Ref Relations: ",ref_relations)
#     st.write("Pred Relations: ",model_relations)
#     st.write("======CHM_END============")
#     # Lengths for easier formula computation
#     len_I = len(inp_tuples)
#     len_R = len(ref_tuples)
#     len_G = len(pred_tuples)

#     # Compute precision, recall, and F1 score
#     precision = len(pred_tuples & ref_tuples) / len(pred_tuples) if len(pred_tuples) > 0 else 0
#     recall = len(pred_tuples & ref_tuples) / len(ref_tuples) if len(ref_tuples) > 0 else 0
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     # Calculate semantic similarity intersections
#     I_intersect_R_sim = calculate_semantic_similarity(inp_tuples, ref_tuples)
#     I_intersect_G_sim = calculate_semantic_similarity(inp_tuples, pred_tuples)
#     R_intersect_G_sim = calculate_semantic_similarity(ref_tuples, pred_tuples)
#     I_intersect_R_intersect_G_sim = calculate_semantic_similarity_three_sets(inp_tuples,ref_tuples,pred_tuples)
#     I_intersect_R_intersect_R_sim = calculate_semantic_similarity_three_sets(inp_tuples,ref_tuples,ref_relations)
    
#     # I_intersect_R_intersect_G_sim = calculate_three_way_semantic_similarity(inp_tuples  ref_tuples, pred_tuples)


#     # Computation of metrics based on the corrected formulas
#     ef = (3 * I_intersect_R_intersect_G_sim) / (len_I + len_R + len_G) if (len_I + len_R + len_G) > 0 else 0
#     ph = (2 * R_intersect_G_sim) / (len_R + len_G) if (len_R + len_G) > 0 else 0
#     of = (2 * I_intersect_G_sim) / (len_I + len_G) if (len_I + len_G) > 0 else 0

#     # Correct NH calculation
#     nh_numerator = len_G - (R_intersect_G_sim + I_intersect_G_sim - I_intersect_R_intersect_G_sim)
#     nh = nh_numerator / len_G if len_G > 0 else 0

#     # Correct LF calculation
#     lf = (len_R - (I_intersect_R_sim - I_intersect_R_intersect_G_sim)) / (len_R + len_G) if (len_R + len_G) > 0 else 0

#     # Lost Hallucination calculation
#     lh_numerator = len_I - I_intersect_G_sim
#     lh = lh_numerator / (len_I + len_G) if (len_I + len_G) > 0 else 0
#     rhi = 3 + (ef * ph) - (of + nh + lf + lh)

#     # Initialize result with computed metrics
#     result = {
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1_score,
#         "ef": ef,
#         "ph": ph,
#         "of": of,
#         "nh": nh,
#         "lf": lf,
#         "Lost Hallucination": lh,
#         "rhi": rhi,
#         "semantic_similarities": {
#             "I_intersect_R_sim": I_intersect_R_sim,
#             "I_intersect_G_sim": I_intersect_G_sim,
#             "R_intersect_G_sim": R_intersect_G_sim,
#             "I_intersect_R_intersect_G_sim": I_intersect_R_intersect_G_sim,
#             "I_intersect_R_intersect_R_sim":I_intersect_R_intersect_R_sim,
#             "lenI":len_I,
#             "lenG":len_G,
#             "lenR":len_R,
#             "lh_numerator":lh_numerator
#         }
#     }

#     # Helper function to replace negative values with 0.001
#     def replace_negatives(value):
#         return max(value,value)

#     # Apply replace_negatives only to semantic_similarities
#     result["semantic_similarities"] = {
#         k: replace_negatives(v) for k, v in result["semantic_similarities"].items()
#     }

#     # Convert to serializable format before returning
#     return convert_to_serializable(result)

# part2 code with some issue of overcount 
# from decimal import Decimal
# import streamlit as st

# Updated the functions of these so this is the old code 
# def calculate_exact_intersection_count(set1, set2):
#     """
#     Calculate the count of exact and partial matches between two sets.
#     Partial matches are weighted as 1/3, 2/3, or 1 based on the number of matching components.
#     """
#     match_count = Decimal(0)
#     matched_tups = set()

#     # Sort sets for consistent order
#     set1 = sorted(set1)
#     set2 = sorted(set2)

#     for tup2 in set2:
#         subject2, verb2, object2 = tup2[:-1]  # Extract SVO components from set2

#         for tup1 in set1:
#             if tup1 in matched_tups:
#                 continue  # Skip already counted matches

#             subject1, verb1, object1 = tup1[:-1]  # Extract SVO components from set1
#             match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

#             if match_elements > 0:
#                 match_count += Decimal(match_elements) / Decimal(3)  # Use Decimal for precision
#                 matched_tups.add(tup1)  # Track matched tuple to avoid double counting
#                 break

#     return float(match_count)

# def calculate_exact_triple_intersection_count(set1, set2, set3):
#     """
#     Calculate the count of exact and partial matches in the intersection of three sets.
#     Partial matches are weighted as 1/3, 2/3, or 1 based on the number of matching components.
#     """
#     match_count = Decimal(0)
#     matched_tups = set()

#     # Sort sets for consistent order
#     set1 = sorted(set1)
#     set2 = sorted(set2)
#     set3 = sorted(set3)

#     for tup3 in set3:
#         subject3, verb3, object3 = tup3[:-1]  # Extract SVO components from set3

#         for tup2 in set2:
#             subject2, verb2, object2 = tup2[:-1]  # Extract SVO components from set2

#             if (subject2, verb2, object2) == (subject3, verb3, object3):  # Exact match with set3
#                 for tup1 in set1:
#                     if tup1 in matched_tups:
#                         continue  # Skip already counted matches

#                     subject1, verb1, object1 = tup1[:-1]  # Extract SVO components from set1
#                     match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

#                     if match_elements > 0:
#                         match_count += Decimal(match_elements) / Decimal(3)
#                         matched_tups.add(tup1)
#                         break

#     return float(match_count)

# from decimal import Decimal
# import streamlit as st

# def calculate_exact_intersection_count(set1, set2):
#     """
#     Calculate the count of exact and partial matches between two sets.
#     Partial matches are weighted as 1/3, 2/3, or 1 based on the number of matching components.
#     """
#     match_count = Decimal(0)
#     matched_tups = set()

#     # Sort sets for consistent order
#     set1 = sorted(set1)
#     set2 = sorted(set2)

#     for tup2 in set2:
#         subject2, verb2, object2 = tup2[:-1]  # Extract SVO components from set2

#         for tup1 in set1:
#             if tup1 in matched_tups:
#                 continue  # Skip already counted matches

#             subject1, verb1, object1 = tup1[:-1]  # Extract SVO components from set1
#             match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

#             if match_elements > 0:
#                 match_count += Decimal(match_elements) / Decimal(3)  # Use Decimal for precision
#                 matched_tups.add(tup1)  # Track matched tuple to avoid double counting
#                 break

#     return float(match_count)

# def calculate_exact_triple_intersection_count(set1, set2, set3):
#     """
#     Calculate the count of exact and partial matches in the intersection of three sets.
#     Partial matches are weighted as 1/3, 2/3, or 1 based on the number of matching components.
#     """
#     match_count = Decimal(0)
#     matched_tups = set()

#     # Sort sets for consistent order
#     set1 = sorted(set1)
#     set2 = sorted(set2)
#     set3 = sorted(set3)

#     # Check if set2 and set3 are identical to avoid double-counting
#     if set2 == set3:
#         for tup2 in set2:
#             subject2, verb2, object2 = tup2[:-1]  # Extract SVO components from set2

#             for tup1 in set1:
#                 if tup1 in matched_tups:
#                     continue  # Skip already counted matches

#                 subject1, verb1, object1 = tup1[:-1]  # Extract SVO components from set1
#                 match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

#                 if match_elements > 0:
#                     match_count += Decimal(match_elements) / Decimal(3)
#                     matched_tups.add(tup1)
#                     break

#     else:
#         # If set2 and set3 are not identical, perform the full calculation
#         for tup3 in set3:
#             subject3, verb3, object3 = tup3[:-1]  # Extract SVO components from set3

#             for tup2 in set2:
#                 subject2, verb2, object2 = tup2[:-1]  # Extract SVO components from set2

#                 if (subject2, verb2, object2) == (subject3, verb3, object3):  # Exact match with set3
#                     for tup1 in set1:
#                         if tup1 in matched_tups:
#                             continue  # Skip already counted matches

#                         subject1, verb1, object1 = tup1[:-1]  # Extract SVO components from set1
#                         match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

#                         if match_elements > 0:
#                             match_count += Decimal(match_elements) / Decimal(3)
#                             matched_tups.add(tup1)
#                             break

#     return float(match_count)



def is_semantically_valid(tup1, tup2):
    """Check semantic similarity using sentence embeddings only for verbs."""
    verb1 = tup1[1]
    verb2 = tup2[1]

    # Compute embeddings for verbs
    embeddings = model.encode([verb1, verb2], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()

    # print(f"Comparing Verbs: '{verb1}' with '{verb2}' | Cosine Similarity: {cosine_sim}")
    return cosine_sim > 0.3


# Global lists to persist results
GLOBAL_MATCHED_WORDS = []
GLOBAL_UNMATCHED_WORDS = []
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer, util

# Load model globally
model = SentenceTransformer('all-MiniLM-L6-v2')


def update_global_lists(matched, unmatched):
    GLOBAL_MATCHED_WORDS.extend(matched)
    GLOBAL_UNMATCHED_WORDS.extend(unmatched)
from decimal import Decimal

def calculate_exact_intersection_count(set1, set2):
    """
    Calculate the count of exact and partial matches between two sets.
    Partial matches are weighted as 1/3, 2/3, or 1 based on the number of matching components.
    """
    match_count = Decimal(0)
    matched_tups = set()
    matched_words = []
    unmatched_words = list(set1)

    set1 = sorted(set1)
    set2 = sorted(set2)

    for tup2 in set2:
        subject2, verb2, object2 = tup2[:-1]

        for tup1 in set1:
            if tup1 in matched_tups:
                continue

            subject1, verb1, object1 = tup1[:-1]
            match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

            if match_elements > 0 and is_semantically_valid(tup1, tup2):
                match_count += Decimal(match_elements) / Decimal(3)
                matched_tups.add(tup1)
                matched_words.append((tup1, tup2))
                if tup1 in unmatched_words:
                    unmatched_words.remove(tup1)
                # print(f"Matched: {tup1} with {tup2} | Match Elements: {match_elements} | Current Count: {float(match_count)}")
                break

    update_global_lists(matched_words, unmatched_words)
    return float(match_count)


def calculate_exact_triple_intersection_count(set1, set2, set3):
    """
    Calculate the count of exact and partial matches in the intersection of three sets.
    Partial matches are weighted as 1/3, 2/3, or 1 based on the number of matching components.
    """
    match_count = Decimal(0)
    matched_tups = set()
    matched_words = []
    unmatched_words = list(set1)

    set1 = sorted(set1)
    set2 = sorted(set2)
    set3 = sorted(set3)

    if set2 == set3:
        for tup2 in set2:
            subject2, verb2, object2 = tup2[:-1]

            for tup1 in set1:
                if tup1 in matched_tups:
                    continue

                subject1, verb1, object1 = tup1[:-1]
                match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

                if match_elements > 0 and is_semantically_valid(tup1, tup2):
                    match_count += Decimal(match_elements) / Decimal(3)
                    matched_tups.add(tup1)
                    matched_words.append((tup1, tup2))
                    if tup1 in unmatched_words:
                        unmatched_words.remove(tup1)
                    # print(f"Matched: {tup1} with {tup2} | Match Elements: {match_elements} | Current Count: {float(match_count)}")
                    break

    else:
        for tup3 in set3:
            subject3, verb3, object3 = tup3[:-1]

            for tup2 in set2:
                subject2, verb2, object2 = tup2[:-1]

                if (subject2, verb2, object2) == (subject3, verb3, object3):
                    for tup1 in set1:
                        if tup1 in matched_tups:
                            continue

                        subject1, verb1, object1 = tup1[:-1]
                        match_elements = sum([subject1 == subject2, verb1 == verb2, object1 == object2])

                        if match_elements > 0 and is_semantically_valid(tup1, tup2):
                            match_count += Decimal(match_elements) / Decimal(3)
                            matched_tups.add(tup1)
                            matched_words.append((tup1, tup2, tup3))
                            if tup1 in unmatched_words:
                                unmatched_words.remove(tup1)
                            # print(f"Matched: {tup1} with {tup2} and {tup3} | Match Elements: {match_elements} | Current Count: {float(match_count)}")
                            break

    update_global_lists(matched_words, unmatched_words)
    return float(match_count)


def get_global_results():
    """Retrieve all matched and unmatched words."""
    return {
        "Matched Words": GLOBAL_MATCHED_WORDS,
        "Unmatched Words": GLOBAL_UNMATCHED_WORDS
    }

import numpy as np

def compute_statistics(metrics_list):
    """
    Compute mean, variance, and standard deviation for a list of metric dictionaries.

    Args:
        metrics_list (list): A list of dictionaries containing metric values.

    Returns:
        dict: A dictionary containing mean, variance, and standard deviation for each metric.
    """
    if not metrics_list:
        return {}

    # Extract metric names
    metric_names = metrics_list[0].keys()
    
    # Initialize result dictionary
    statistics = {}

    for metric in metric_names:
        if metric == "intersection_counts":  # Skip intersection counts
            continue
            
        values = np.array([entry[metric] for entry in metrics_list])
        
        statistics[metric] = {
            "mean": np.mean(values),
            "variance": np.var(values),
            "std_dev": np.std(values)
        }

    return statistics

import numpy as np

def compute_rhi(PH, EF, NH, OF, LF, LH):
    numerator = 1 + PH + EF
    denominator = 1 + PH + EF + np.sqrt(NH + OF + LF + LH)  # Soft penalization
    
    # Handling zero-case
    if numerator == 1 and denominator == 1:
        return 0  # Return 0 instead of 1
    
    return numerator / denominator

def compute_hallucination_metrics(pred_words, ref_words, inp_words, input_relations, ref_relations, model_relations):
    # Filter symbols and get tuple counts
    pred_tuples = filter_unwanted_symbols(set(model_relations))
    ref_tuples = filter_unwanted_symbols(set(ref_relations))
    inp_tuples = filter_unwanted_symbols(set(input_relations))
    
    # Set lengths for easy formula use
    len_I = len(inp_tuples)
    len_R = len(ref_tuples)
    len_G = len(pred_tuples)

   

    # Calculate intersections
    I_intersect_R_count = calculate_exact_intersection_count(inp_tuples, ref_tuples)
    I_intersect_G_count = calculate_exact_intersection_count(inp_tuples, pred_tuples)
    R_intersect_G_count = calculate_exact_intersection_count(ref_tuples, pred_tuples)
    I_intersect_R_intersect_G_count = calculate_exact_triple_intersection_count(inp_tuples, ref_tuples, pred_tuples)
    
     # Precision, recall, F1
    precision = I_intersect_G_count / len_G if len_G > 0 else 0
    recall = R_intersect_G_count / len_R if len_R > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Hallucination metrics with capped values
    ef = (3 * I_intersect_R_intersect_G_count) / (len_I + len_R + len_G) if (len_I + len_R + len_G) > 0 else 0
    ph = (2 * R_intersect_G_count) / (len_R + len_G) if (len_R + len_G) > 0 else 0
    of = (2 * (I_intersect_G_count-I_intersect_R_intersect_G_count)) / (len_I + len_G) if (len_I + len_G) > 0 else 0

    nh_numerator = (len_G - (R_intersect_G_count + I_intersect_G_count - I_intersect_R_intersect_G_count))
    nh = abs(nh_numerator) / len_G if len_G > 0 else 0
    nh = abs(nh)
    # lf_numerator = len_R - (I_intersect_R_count - I_intersect_R_intersect_G_count)
    # lf = lf_numerator / (len_R + len_G) if (len_R + len_G) > 0 else 0

    # lh_numerator =  len_I - I_intersect_G_count
    # lh = lh_numerator / (len_I + len_G) if (len_I + len_G) > 0 else 0
    
    
    #Lost Focus:
    # lf = I_intersect_R_count - I_intersect_R_intersect_G_count) / len_G
    
    lf_numerator = (I_intersect_R_count -  I_intersect_R_intersect_G_count)
    lf = lf_numerator / (len_G) if (len_G) > 0 else 0
    
    # lf_numerator = len_R - (I_intersect_R_count - I_intersect_R_intersect_G_count)
    # lf = lf_numerator / (len_R + len_G) if (len_R + len_G) > 0 else 0
    
    
    # LH=2 *(R-(I INTER R) -((R INTER G) - (I INTER R INTER G)))/ R+G 

    # lh_numerator =  len_I - I_intersect_G_count
    
    lh_numerator =  2*(len_R-(I_intersect_R_count)-((R_intersect_G_count)-(I_intersect_R_intersect_G_count)))
    
    lh = lh_numerator / (len_I + len_G) if (len_I + len_G) > 0 else 0


    # rhi = 2 + (ef * ph) - (of + nh + lf + lh)
    
    # rhi = 1+( (ef *ph)/2) - ( (of + lh + lf + nh)/4 )
    
    rhi = compute_rhi(ph, ef, nh, of, lf, lh)
    

    result = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "ef": ef,
        "ph": ph,
        "of": of,
        "nh": nh,
        "lf": lf,
        "Lost Hallucination": lh,
        "rhi": rhi,
        "intersection_counts": {
            "I_intersect_R_count": I_intersect_R_count,
            "I_intersect_G_count": I_intersect_G_count,
            "R_intersect_G_count": R_intersect_G_count,
            "I_intersect_R_intersect_G_count": I_intersect_R_intersect_G_count,
            "lenI": len_I,
            "lenG": len_G,
            "lenR": len_R,
            "lh_numerator": lh_numerator,
            "nh_numerator":nh_numerator
        }
    }
    

    return convert_to_serializable(result)


def convert_svo_to_tuples(svo_relations):
    # Convert confidence to string to avoid type issues
    return [
        (rel['subject'], rel['verb'], rel['object'], str(rel['confidence'])) 
        for rel in svo_relations
    ]

def process_json_and_extract_relations(json_data):
    # Check if the data is already processed to avoid reprocessing
    if 'processed_data' in st.session_state:
        return st.session_state.processed_data

    progress_bar = st.progress(0)
    total_entries = len(json_data)

    processed_data = {}
    for i, (index, entry) in enumerate(json_data.items()):
        try:
            if 'InputText' in entry:
                # Extract SVO relations for InputText and ReferenceSummary
                entry['svo_relations_input'] = extract_relations_with_confidence(entry.get('InputText', ''))
                entry['svo_relations_ref'] = extract_relations_with_confidence(entry.get('ReferenceSummary', ''))

                # Process each model
                models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
                for model_name in models:
                    summary = entry.get(model_name, '')

                    # Extract SVO relations for each model's summary
                    entry[f"svo_relations_{model_name}"] = extract_relations_with_confidence(summary)

                    # Tokenize text
                    pred_words = nltk.word_tokenize(summary)
                    ref_words = nltk.word_tokenize(entry.get('ReferenceSummary', ''))
                    inp_words = nltk.word_tokenize(entry.get('InputText', ''))

                    # Compute ROUGE scores
                    rouge_scores = compute_rouge(summary, entry.get('ReferenceSummary', ''))
                    entry[f"{model_name}_rouge"] = rouge_scores
                    
                    # Compute average ROUGE scores
                    rouge_scores_02 = compute_rouge_average(rouge_scores)
                    entry[f"{model_name}_rouge_avg"] = rouge_scores_02

                    # Get filtered relations for hallucination metrics
                    input_relations = convert_svo_to_tuples(entry['svo_relations_input'])
                    ref_relations = convert_svo_to_tuples(entry['svo_relations_ref'])
                    model_relations = convert_svo_to_tuples(entry[f"svo_relations_{model_name}"])

                    # Compute hallucination metrics, including intermediate data
                    hallucination_metrics = compute_hallucination_metrics(pred_words, ref_words, inp_words,input_relations, ref_relations, model_relations)
                    entry[f"{model_name}_hallucination_metrics"] = hallucination_metrics

            # Store the processed entry
            processed_data[index] = entry

        except KeyError as e:
            st.write(f"KeyError: {e} in entry with index: {index}")
        except Exception as e:
            st.write(f"An error occurred: {e}")

        # Update progress bar
        progress_bar.progress((i + 1) / total_entries)

    st.session_state.processed_data = processed_data
    return processed_data

import json
import numpy as np

import numpy as np
import torch

def convert_to_serializable(obj):
    """
    Recursively converts non-serializable types (like float32, int32, and PyTorch tensors) to serializable types.
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        # Convert PyTorch tensors to lists
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list elements
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        # Convert tuple elements
        return tuple(convert_to_serializable(i) for i in obj)
    else:
        # Return the object as is if it doesn't need conversion
        return obj

def save_json_to_string(data):
    """
    Converts the given data into a JSON serializable string.
    """
    try:
        serializable_data = convert_to_serializable(data)  # Convert any non-serializable data
        json_str = json.dumps(serializable_data, indent=4)
        return json_str
    except TypeError as e:
        print(f"Serialization error: {e}")
        raise

# Example usage in your main function
# json_str = save_json_to_string(st.session_state.processed_data)


def plot_rouge_scores(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rouge_types = ['rouge1', 'rouge2', 'rougeL']

    # Initialize DataFrames to store raw and aggregated ROUGE scores
    df_raw_rouge_scores = pd.DataFrame()
    df_rouge_scores = pd.DataFrame(columns=['Model'] + rouge_types)

    # Iterate over each model
    for model_name in models:
        # Create an empty list to hold all ROUGE scores
        all_rouge_scores = {rouge_type: [] for rouge_type in rouge_types}
        detailed_scores = []

        # Collect ROUGE scores for each type for the current model
        for i in range(len(data)):
            index_str = str(i)
            model_key = f"{model_name}_rouge"

            # Check if the model_key exists in the data at index i
            if index_str in data and model_key in data[index_str]:
                rouge_data = data[index_str][model_key]

                # Collect detailed ROUGE scores for each type
                detailed_row = {'Index': i, 'Model': model_name}
                for rouge_type in rouge_types:
                    scores = rouge_data.get(rouge_type, [])
                    if scores:  # Ensure that scores is a non-empty list
                        detailed_row[rouge_type] = np.mean(scores)
                        all_rouge_scores[rouge_type].append(np.mean(scores))
                    else:
                        detailed_row[rouge_type] = np.nan
                        all_rouge_scores[rouge_type].append(np.nan)

                detailed_scores.append(detailed_row)

        # Append detailed scores to the DataFrame
        df_raw_rouge_scores = pd.concat([df_raw_rouge_scores, pd.DataFrame(detailed_scores)], ignore_index=True)

        # Aggregate ROUGE scores by calculating the mean
        rouge_scores_agg = {
            rouge_type: np.nanmean(all_rouge_scores[rouge_type])
            for rouge_type in rouge_types
        }

        # Append the results to the DataFrame
        df_rouge_scores = pd.concat([df_rouge_scores, pd.DataFrame([{
            'Model': model_name,
            'rouge1': rouge_scores_agg['rouge1'],
            'rouge2': rouge_scores_agg['rouge2'],
            'rougeL': rouge_scores_agg['rougeL']
        }])], ignore_index=True)

        # Plotting the ROUGE scores
        plt.figure(figsize=(10, 6))
        plt.bar(rouge_scores_agg.keys(), rouge_scores_agg.values(), color='skyblue')
        plt.title(f'ROUGE Scores for {model_name}')
        plt.xlabel('ROUGE Type')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        st.pyplot(plt.gcf())
        plt.clf()

    # Display the raw ROUGE scores in a table
    st.write("Raw ROUGE Scores")
    st.dataframe(df_raw_rouge_scores)

    # Display the aggregated ROUGE scores in a table
    st.write("Aggregated ROUGE Scores")
    st.table(df_rouge_scores)

# def analyze_graph(df, rouge_type, metric, model_name):
#     correlation = df[rouge_type].corr(df[metric])
#     analysis = ""
    
#     if correlation > 0.7:
#         analysis = f"For {model_name}, there is a strong positive correlation ({correlation:.2f}) between {rouge_type} and {metric}, indicating that as {rouge_type} increases, {metric} tends to increase as well."
#     elif correlation < -0.7:
#         analysis = f"For {model_name}, there is a strong negative correlation ({correlation:.2f}) between {rouge_type} and {metric}, indicating that as {rouge_type} increases, {metric} tends to decrease."
#     elif -0.3 <= correlation <= 0.3:
#         analysis = f"For {model_name}, there is little to no correlation ({correlation:.2f}) between {rouge_type} and {metric}, suggesting they vary independently."
#     else:
#         analysis = f"For {model_name}, there is a moderate correlation ({correlation:.2f}) between {rouge_type} and {metric}, showing some degree of linear relationship."

#     # Check for outliers
#     rouge_outliers = df[rouge_type][(df[rouge_type] > df[rouge_type].mean() + 2 * df[rouge_type].std()) | (df[rouge_type] < df[rouge_type].mean() - 2 * df[rouge_type].std())]
#     metric_outliers = df[metric][(df[metric] > df[metric].mean() + 2 * df[metric].std()) | (df[metric] < df[metric].mean() - 2 * df[metric].std())]
    
#     if len(rouge_outliers) > 0 or len(metric_outliers) > 0:
#         analysis += f" Notable outliers were observed, which may indicate specific cases where the model's performance deviates from the general trend."

#     return analysis
# def plot_correlation_analysis_with_report(data):
#     models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo"]
#     metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
#     rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
#     colors = {'rouge': 'blue', 'hallucination': 'red'}
    
#     for model_name in models:
#         metrics_data = {metric: [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)] for metric in metrics}
#         rouge_scores = {rouge_type: [data[str(i)][f"{model_name}_rouge"][rouge_type].fmeasure for i in range(400)] for rouge_type in rouge_types}
        
#         # Create DataFrame for correlation analysis
#         df = pd.DataFrame(rouge_scores)
#         df = df.join(pd.DataFrame(metrics_data))
        
#         # Line plot of ROUGE vs. hallucination metrics with two colors and index labels
#         for rouge_type in rouge_types:
#             for metric in metrics:
#                 plt.figure(figsize=(12, 8))
                
#                 # Plot the line for ROUGE
#                 plt.plot(df[rouge_type], label=f'{rouge_type} (ROUGE)', color=colors['rouge'], linestyle='-', marker='o', markersize=3)
                
#                 # Plot the line for the hallucination metric
#                 plt.plot(df[metric], label=f'{metric} (Hallucination)', color=colors['hallucination'], linestyle='-', marker='o', markersize=3)
                
#                 # Add small index labels for all points
#                 for i in range(len(df)):
#                     plt.text(i, df[rouge_type].iloc[i], str(i), fontsize=6, ha='right', color='gray', alpha=0.7)
#                     plt.text(i, df[metric].iloc[i], str(i), fontsize=6, ha='left', color='gray', alpha=0.7)
                
#                 plt.title(f'{model_name}: {rouge_type} vs. {metric}')
#                 plt.xlabel('Index')
#                 plt.ylabel('Scores')
#                 plt.legend(loc="best")
#                 plt.grid(True)
#                 st.pyplot(plt.gcf())
#                 plt.clf()
                
#                 # Generate analysis report
#                 analysis = analyze_graph(df, rouge_type, metric, model_name)
#                 st.write(analysis)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_metrics_with_analysis(processed_data):
    if not processed_data:
        st.error("No processed data provided.")
        return
    
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    colors = {'rouge': 'blue', 'hallucination': 'red'}

    all_data = []

    for model_name in models:
        metrics_data = {metric: [] for metric in metrics}
        rouge_scores = {rouge_type: [] for rouge_type in rouge_types}
        indices = []

        for index in processed_data.keys():
            entry = processed_data[index]
            indices.append(index)

            model_hallucination_key = f"{model_name}_hallucination_metrics"
            if model_hallucination_key in entry:
                hallucination_metrics = entry[model_hallucination_key]
                for metric in metrics:
                    metrics_data[metric].append(hallucination_metrics.get(metric, np.nan))
            else:
                for metric in metrics:
                    metrics_data[metric].append(np.nan)

            model_rouge_key = f"{model_name}_rouge_avg"
            if model_rouge_key in entry:
                rouge_avg = entry[model_rouge_key]
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type].append(rouge_avg.get(f"avg_{rouge_type}", np.nan))
            else:
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type].append(np.nan)

        df = pd.DataFrame(rouge_scores).join(pd.DataFrame(metrics_data))
        df.index = indices
        df = df.apply(pd.to_numeric, errors='coerce')

        st.write(f"DataFrame for {model_name}:")
        st.dataframe(df)

        # Determine the interval size based on the number of indexes
        num_indexes = len(df.index)
        if num_indexes > 100:
            interval = max(1, num_indexes // 10)  # For large datasets, show intervals of 10%
        else:
            interval = 1  # For smaller datasets, show each index

        x_ticks = list(range(0, num_indexes, interval))
        x_tick_labels = [f'{i}-{min(i + interval - 1, num_indexes - 1)}' for i in x_ticks]

        for rouge_type in rouge_types:
            for metric in metrics:
                st.markdown(f"### Graph for {model_name}: {rouge_type} vs. {metric}")

                plt.figure(figsize=(14, 10))  # Increase figure size

                plt.plot(df.index, df[rouge_type], label=f'{rouge_type} (ROUGE)', color=colors['rouge'], linestyle='-', marker='o', markersize=5)
                plt.plot(df.index, df[metric], label=f'{metric} (Hallucination)', color=colors['hallucination'], linestyle='-', marker='o', markersize=5)

                # Compute correlation
                correlation = df[rouge_type].corr(df[metric])
                analysis = f"**Analysis of {model_name}: {rouge_type} vs. {metric}**\n\n"
                if correlation > 0.7:
                    analysis += f"There is a strong positive correlation ({correlation:.2f}) between {rouge_type} and {metric}."
                elif correlation < -0.7:
                    analysis += f"There is a strong negative correlation ({correlation:.2f}) between {rouge_type} and {metric}."
                elif -0.3 <= correlation <= 0.3:
                    analysis += f"There is little to no correlation ({correlation:.2f}) between {rouge_type} and {metric}."
                else:
                    analysis += f"There is a moderate correlation ({correlation:.2f}) between {rouge_type} and {metric}."

                st.markdown(analysis)

                plt.title(f'{model_name}: {rouge_type} vs. {metric}')
                plt.xlabel('Index')
                plt.ylabel('Scores')

                # Set x-ticks and labels
                plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=0, fontsize=10)
                plt.yticks(fontsize=10)
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()  # Adjust layout to fit everything
                st.pyplot(plt.gcf())
                plt.clf()

                st.markdown("---")

        if all_data:
            combined_df = pd.concat(all_data, keys=models, names=['Model', 'Index'])
            correlation_matrix = combined_df.corr()

            st.write("### Correlation Matrix for All Models")
            st.dataframe(correlation_matrix)



import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def plot_hallucination_metrics_indexwise(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo", "RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf','rhi']
    model_scores = {model: [] for model in models}  # To store scores for comparison

    for model_name in models:
        # Dynamically determine data length from the provided data
        data_length = max(int(idx) for idx in data.keys()) + 1
        
        # Prepare the data for each model, handling missing data with .get()
        metrics_data = {metric: [
            data.get(str(i), {}).get(f"{model_name}_hallucination_metrics", {}).get(metric, 0) 
            for i in range(data_length)
        ] for metric in metrics}

        # Convert metrics_data to a DataFrame for easier plotting
        df = pd.DataFrame(metrics_data)
        
        for metric in metrics:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(data_length)),
                y=df[metric],
                mode='lines+markers',
                marker=dict(size=6, color='lightcoral'),
                line=dict(color='lightcoral'),
                name=metric,
                hovertemplate="Index: %{x}<br>Value: %{y:.2f}"  # Show index and metric value on hover
            ))

            fig.update_layout(
                title=f'Index-wise Line Plot of {metric.capitalize()} for {model_name}',
                xaxis_title="Index",
                yaxis_title=metric.capitalize(),
                template="plotly_white",
                height=500,
                margin=dict(l=0, r=0, t=40, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate summary statistics
            mean_value = df[metric].mean()
            median_value = df[metric].median()
            std_dev = df[metric].std()
            
            # Update model scores
            model_scores[model_name].append(mean_value)
            
            # Display the report in Streamlit
            st.write(f"### Hallucination Report for {metric} - {model_name}")
            st.write(f"**Mean {metric.capitalize()}:** {mean_value:.2f}")
            st.write(f"**Median {metric.capitalize()}:** {median_value:.2f}")
            st.write(f"**Standard Deviation of {metric.capitalize()}:** {std_dev:.2f}")
            
            # Provide an analysis based on the graph
            st.write(f"**Analysis of {metric}:**")
            if df[metric].mean() > 0.5:
                st.write(f"The average {metric} value is relatively high, indicating a trend of significant {metric} across indices.")
            else:
                st.write(f"The average {metric} value is relatively low, suggesting less pronounced {metric} in the data.")
            
            # Describe any observed patterns
            if df[metric].max() > 0.8:
                st.write(f"There are noticeable peaks in {metric} values, suggesting certain indices have high {metric}.")
            if df[metric].min() < 0.2:
                st.write(f"Some indices show very low {metric}, indicating variability in {metric} across the dataset.")
                
            st.write("---")
    
    # Calculate overall hallucination scores for each model
    overall_scores = {model: sum(scores)/len(scores) for model, scores in model_scores.items()}
    
    # Identify the model with the highest overall hallucination score
    max_model = max(overall_scores, key=overall_scores.get)
    
    # Display a generic report
    st.write("## Generic Hallucination Report")
    st.write(f"Based on the analysis, the model with the highest average hallucination score is **{max_model}**.")
    st.write("### Predictions for Each Model:")
    
    for model, score in overall_scores.items():
        if model == max_model:
            st.write(f"**{model}:** This model tends to generate the highest hallucination scores, indicating it may be more prone to generating irrelevant or inaccurate information.")
        else:
            st.write(f"**{model}:** This model has lower average hallucination scores, suggesting it might be better at maintaining relevant and accurate information compared to {max_model}.")
    
    st.write("---")


import matplotlib.pyplot as plt
import streamlit as st

def plot_hallucination_metrics(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    
    # Convert keys to a list of valid indices (assume they might not be sequential or strings)
    available_indices = [int(idx) for idx in data.keys() if str(idx).isdigit()]
    available_indices.sort()  # Sort indices for better plotting

    for model_name in models:
        metrics_data = {}
        
        # Populate metrics data for available indices
        for metric in metrics:
            metrics_data[metric] = [
                data.get(str(i), {}).get(f"{model_name}_hallucination_metrics", {}).get(metric, 0) 
                for i in available_indices
            ]
        
        # Plot histograms for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.hist(metrics_data[metric], bins=20, color='lightcoral', edgecolor='black')
            plt.title(f'Distribution of {metric} for {model_name}')
            plt.xlabel(metric.capitalize())
            plt.ylabel('Frequency')
            plt.grid(axis='y')
            st.pyplot(plt.gcf())
            plt.clf()

# Function to plot CDF graphs for metrics
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_cdf_graphs(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf', 'rhi']  # Added 'rhi'
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for model_name in models:
            metric_values = []
            for i in range(400):
                model_key = f"{model_name}_hallucination_metrics"
                if model_key in data.get(str(i), {}):
                    metric_value = data[str(i)][model_key].get(metric, None)
                    if isinstance(metric_value, list):
                        metric_value = np.mean(metric_value)  # Aggregate list values
                    if metric_value is not None:
                        metric_values.append(metric_value)
            
            if metric_values:
                # Sort and plot CDF
                sorted_values = np.sort(metric_values)
                cdf = np.arange(len(sorted_values)) / float(len(sorted_values))
                plt.plot(sorted_values, cdf, label=model_name)
            else:
                st.write(f"No valid {metric} data to plot for model: {model_name}")

        plt.title(f'Cumulative Distribution Function (CDF) of {metric.capitalize()}')
        plt.xlabel(metric.capitalize())
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()
        
        
        
        
# Function to perform Correlation Analysis
# def plot_correlation_analysis(data):
#     models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo"]
#     metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    
#     for model_name in models:
#         metrics_data = {metric: [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)] for metric in metrics}
#         rouge_scores = {rouge_type: [data[str(i)][f"{model_name}_rouge"][rouge_type].fmeasure for i in range(400)] for rouge_type in ['rouge1', 'rouge2', 'rougeL']}
        
#         # Create DataFrame for correlation analysis
#         df = pd.DataFrame(rouge_scores)
#         df = df.join(pd.DataFrame(metrics_data))
        
#         # Compute and plot correlation matrix
#         corr_matrix = df.corr()
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#         plt.title(f'Correlation Matrix for {model_name}')
#         st.pyplot(plt.gcf())
#         plt.clf()
        
#         # Scatter plot of ROUGE vs. hallucination metrics with index and labels
#         for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
#             for metric in metrics:
#                 plt.figure(figsize=(12, 8))
#                 plt.scatter(df[rouge_type], df[metric], alpha=0.5, label=metric)
                
#                 # Add index to each point
#                 for i in range(len(df)):
#                     plt.text(df[rouge_type].iloc[i], df[metric].iloc[i], str(i), fontsize=8, alpha=0.7)

#                 plt.title(f'{rouge_type} vs. {metric}')
#                 plt.xlabel(rouge_type)
#                 plt.ylabel(metric)
#                 plt.legend(title="Metrics")
#                 plt.grid(True)
#                 st.pyplot(plt.gcf())
#                 plt.clf()
                
def plot_correlation_analysis(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    colors = {'rouge': 'darkblue', 'hallucination': 'yellow'}
    
    for model_name in models:
        metrics_data = {metric: [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)] for metric in metrics}
        rouge_scores = {rouge_type: [data[str(i)][f"{model_name}_rouge"][rouge_type].fmeasure for i in range(400)] for rouge_type in rouge_types}
        
        # Create DataFrame for correlation analysis
        df = pd.DataFrame(rouge_scores)
        df = df.join(pd.DataFrame(metrics_data))
        
        # Compute and plot correlation matrix
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation Matrix for {model_name}')
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Scatter plot of ROUGE vs. hallucination metrics with two colors
        for rouge_type in rouge_types:
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                
                # Plot all points with different colors
                plt.scatter(df[rouge_type], df[metric], alpha=0.5, color=colors['rouge'], label=f'{rouge_type} (ROUGE)', s=20)
                plt.scatter(df[rouge_type], df[metric], alpha=0.5, color=colors['hallucination'], label=f'{metric} (Hallucination)', s=20)
                
                # Increase marker size for selected points (e.g., every 20th point)
                plt.scatter(df[rouge_type][::20], df[metric][::20], alpha=0.7, color=colors['hallucination'], s=60)
                
                plt.title(f'{rouge_type} vs. {metric}')
                plt.xlabel(rouge_type)
                plt.ylabel(metric)
                plt.legend(loc="best")
                plt.grid(True)
                st.pyplot(plt.gcf())
                plt.clf()
                
def plot_correlation_analysis_lineplot(processed_data):
    if not processed_data:
        st.error("No processed data provided.")
        return
    
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    colors = {'rouge': 'blue', 'hallucination': 'red', 'avg_rouge': 'green'}
    
    for model_name in models:
        metrics_data = {metric: [] for metric in metrics}
        rouge_scores = {rouge_type: [] for rouge_type in rouge_types}
        avg_rouge_scores = {rouge_type: [] for rouge_type in rouge_types}
        indices = []
        
        for i in range(400):
            index_str = str(i)
            indices.append(index_str)
            
            if index_str in processed_data:
                entry = processed_data[index_str]
                
                # Collect hallucination metrics
                model_hallucination_key = f"{model_name}_hallucination_metrics"
                if model_hallucination_key in entry:
                    hallucination_metrics = entry[model_hallucination_key]
                    for metric in metrics:
                        metrics_data[metric].append(hallucination_metrics.get(metric, np.nan))
                else:
                    for metric in metrics:
                        metrics_data[metric].append(np.nan)

                # Collect ROUGE scores
                model_rouge_key = f"{model_name}_rouge"
                if model_rouge_key in entry:
                    for rouge_type in rouge_types:
                        rouge_list = entry[model_rouge_key].get(rouge_type, [np.nan])
                        # Assuming the first score in the list is the relevant one
                        rouge_score = rouge_list[0] if len(rouge_list) > 0 else np.nan
                        rouge_scores[rouge_type].append(rouge_score)
                else:
                    for rouge_type in rouge_types:
                        rouge_scores[rouge_type].append(np.nan)
                    
                model_rouge_avg_key = f"{model_name}_rouge_avg"
                if model_rouge_avg_key in entry:
                    for rouge_type in rouge_types:
                        avg_rouge_scores[rouge_type].append(entry[model_rouge_avg_key].get(f"avg_{rouge_type}", np.nan))
                else:
                    for rouge_type in rouge_types:
                        avg_rouge_scores[rouge_type].append(np.nan)
            else:
                # Handle missing index gracefully
                for metric in metrics:
                    metrics_data[metric].append(np.nan)
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type].append(np.nan)
                    avg_rouge_scores[rouge_type].append(np.nan)

        # Create DataFrame for analysis
        df = pd.DataFrame(rouge_scores).join(pd.DataFrame(metrics_data))
        df_avg = pd.DataFrame(avg_rouge_scores)
        df.index = indices
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Compute and plot correlation matrix
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation Matrix for {model_name}')
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Determine the interval size based on the number of indexes
        num_indexes = len(df.index)
        if num_indexes > 400:
            interval = max(1, num_indexes // 20)  # For very large datasets, show intervals of ~5%
        elif num_indexes > 200:
            interval = max(1, num_indexes // 10)  # For large datasets, show intervals of ~10%
        elif num_indexes > 100:
            interval = max(1, num_indexes // 5)   # For medium datasets, show intervals of ~20%
        elif num_indexes > 50:
            interval = max(1, num_indexes // 4)   # For smaller datasets, show intervals of ~25%
        else:
            interval = 1  # For very small datasets, show each index
        
        x_ticks = list(range(0, num_indexes, interval))
        x_tick_labels = [f'{i}-{min(i + interval - 1, num_indexes - 1)}' for i in x_ticks]

        # Line plot of ROUGE vs. hallucination metrics with two colors and index labels
        for rouge_type in rouge_types:
            for metric in metrics:
                plt.figure(figsize=(14, 10))
                
                # Plot the line for ROUGE scores
                plt.plot(df.index, df[rouge_type], label=f'{rouge_type} (ROUGE)', color=colors['rouge'], linestyle='-', marker='o', markersize=5)
                
                # Plot the line for the hallucination metric
                plt.plot(df.index, df[metric], label=f'{metric} (Hallucination)', color=colors['hallucination'], linestyle='-', marker='o', markersize=5)
                
                # Plot the average ROUGE scores
                plt.plot(df.index, df_avg[rouge_type], label=f'{rouge_type} Avg. ROUGE', color=colors['avg_rouge'], linestyle='--', marker='x', markersize=4)

                # Add index labels based on interval
                for i in range(len(df)):
                    if i % interval == 0 or num_indexes <= 20:  # Label every interval or if dataset is very small
                        plt.text(i, df[rouge_type].iloc[i], str(i), fontsize=6, ha='right', color='gray', alpha=0.7)
                        plt.text(i, df[metric].iloc[i], str(i), fontsize=6, ha='left', color='gray', alpha=0.7)
                
                plt.title(f'{model_name}: {rouge_type} vs. {metric}')
                plt.xlabel('Index')
                plt.ylabel('Scores')
                plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=90, fontsize=10)  # Adjust x-ticks
                plt.yticks(fontsize=10)
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()  # Adjust layout to fit everything
                st.pyplot(plt.gcf())
                plt.clf()
                
                st.markdown("---")
                
                

def plot_correlation_analysis2(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    colors = {'rouge': 'blue', 'hallucination': 'red'}
    
    for model_name in models:
        metrics_data = {metric: [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)] for metric in metrics}
        rouge_scores = {rouge_type: [data[str(i)][f"{model_name}_rouge"][rouge_type].fmeasure for i in range(400)] for rouge_type in rouge_types}
        
        # Create DataFrame for correlation analysis
        df = pd.DataFrame(rouge_scores)
        df = df.join(pd.DataFrame(metrics_data))
        
        # Compute and plot correlation matrix
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation Matrix for {model_name}')
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Scatter plot of ROUGE vs. hallucination metrics with two colors and index labels
        for rouge_type in rouge_types:
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                
                # Plot all points with different colors
                plt.scatter(df[rouge_type], df[metric], alpha=0.5, color=colors['rouge'], label=f'{rouge_type} (ROUGE)', s=20)
                plt.scatter(df[rouge_type], df[metric], alpha=0.5, color=colors['hallucination'], label=f'{metric} (Hallucination)', s=20)
                
                # Add small index labels for all points
                for i in range(len(df)):
                    plt.text(df[rouge_type].iloc[i], df[metric].iloc[i], str(i), fontsize=6, ha='right', color='gray', alpha=0.7)
                
                plt.title(f'{rouge_type} vs. {metric}')
                plt.xlabel(rouge_type)
                plt.ylabel(metric)
                plt.legend(loc="best")
                plt.grid(True)
                st.pyplot(plt.gcf())
                plt.clf()
          
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from pandas.plotting import table
import io

def dataframe_to_image(df):
    """Convert a DataFrame to an image."""
    fig, ax = plt.subplots(figsize=(12, 6))  # Set the size of the figure
    ax.axis('off')  # Hide the axis

    # Create a table from the DataFrame and add it to the axis
    table(ax, df, loc='center', cellLoc='center', colWidths=[0.2] * len(df.columns))

    # Save the table as an image
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    image_stream.seek(0)
    return image_stream

def plot_statistical_analysis(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    
    # Initialize a list to hold the rows of the summary
    summary_rows = []
    
    for model_name in models:
        # Initialize a dictionary to hold metric values for the current model
        metrics_data = {metric: [] for metric in metrics}
        
        # Collect metric values
        for i in range(len(data)):  # Use dynamic length of data
            index_str = str(i)
            model_key = f"{model_name}_hallucination_metrics"
            
            if index_str in data and model_key in data[index_str]:
                model_metrics = data[index_str][model_key]
                for metric in metrics:
                    if metric in model_metrics:
                        metrics_data[metric].append(model_metrics[metric])
                    else:
                        metrics_data[metric].append(np.nan)  # Handle missing metrics
        
        # Calculate and plot mean and standard deviation for each metric
        for metric in metrics:
            values = metrics_data[metric]
            mean_val = np.nanmean(values)  # Use nanmean to handle NaNs
            std_val = np.nanstd(values)  # Use nanstd to handle NaNs
            
            # Plot Mean and Standard Deviation Bar Chart
            plt.figure(figsize=(10, 6))
            plt.bar(['Mean', 'Standard Deviation'], [mean_val, std_val], color=['skyblue', 'salmon'])
            plt.title(f'Statistical Analysis of {metric} for {model_name}')
            plt.ylabel('Value')
            plt.grid(axis='y')
            st.pyplot(plt.gcf())
            plt.clf()
            
            # Append the statistics to the summary list
            summary_rows.append({
                'Model': model_name,
                'Metric': metric,
                'Mean': mean_val,
                'Standard Deviation': std_val
            })
    
    # Convert the list to a DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Style the DataFrame
    styled_df = summary_df.style.format({
        'Mean': '{:.2f}',
        'Standard Deviation': '{:.2f}'
    }).background_gradient(subset=['Mean', 'Standard Deviation'], cmap='coolwarm')
    
    # Display the styled statistical summary table
    st.subheader('Statistical Summary of Metrics for All Models')
    st.dataframe(styled_df)

    # Convert the DataFrame to an image
    image_stream = dataframe_to_image(summary_df)
    
    # Provide a download link for the image
    st.download_button(
        label="Download Table as Image",
        data=image_stream,
        file_name="statistical_summary_table.png",
        mime="image/png"
    )
    
    # Additional Plot 2: Line Plots for Mean Values
    mean_values = {model_name: {metric: np.nanmean([data.get(str(i), {}).get(f"{model_name}_hallucination_metrics", {}).get(metric, np.nan) for i in range(len(data))]) for metric in metrics} for model_name in models}
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for model_name in models:
            plt.plot(list(mean_values[model_name].keys()), list(mean_values[model_name].values()), marker='o', label=model_name)
        plt.title(f'Mean {metric} Across Models')
        plt.xlabel('Models')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()
    
    # Additional Plot 3: Histograms
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for model_name in models:
            all_values = [data.get(str(i), {}).get(f"{model_name}_hallucination_metrics", {}).get(metric, np.nan) for i in range(len(data))]
            plt.hist(all_values, bins=20, alpha=0.5, label=model_name)
        plt.title(f'Histogram of {metric} Across Models')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()
        
        
# Function to create combined plots for ROUGE scores and hallucination metrics
def plot_combined_rouge_and_metrics(data):
    
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics = ['precision', 'recall', 'f1_score', 'ef', 'ph', 'of', 'nh', 'lf']
    
    for model_name in models:
        plt.figure(figsize=(14, 10))
        
        # Plot ROUGE scores
        plt.subplot(2, 1, 1)
        rouge_scores = [data[str(i)][f"{model_name}_rouge"] for i in range(400)]
        rouge_scores_agg = {rouge_type: np.mean([score[rouge_type].fmeasure for score in rouge_scores]) for rouge_type in ['rouge1', 'rouge2', 'rougeL']}
        plt.bar(rouge_scores_agg.keys(), rouge_scores_agg.values(), color='skyblue')
        plt.title(f'ROUGE Scores for {model_name}')
        plt.xlabel('ROUGE Type')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        
        # Plot Hallucination metrics
        plt.subplot(2, 1, 2)
        metrics_data = {metric: [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)] for metric in metrics}
        
        for metric in metrics:
            sns.histplot(metrics_data[metric], bins=20, label=metric, kde=True)
            
        plt.title(f'Distribution of Hallucination Metrics for {model_name}')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        st.pyplot(plt.gcf())
        plt.clf()
import seaborn as sns
import base64 
def plot_all_metrics_indexwise(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    metrics_split_1 = ['precision', 'recall', 'f1_score', 'ef']
    metrics_split_2 = ['ph', 'of', 'nh', 'lf']
    
    # Create the first figure
    plt.figure(figsize=(100, 20))
    sns.set(style="whitegrid")
    colors = sns.color_palette("husl", n_colors=len(models) * len(metrics_split_1))
    
    for model_idx, model_name in enumerate(models):
        for metric_idx, metric in enumerate(metrics_split_1):
            metric_data = [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)]
            plt.plot(metric_data, label=f'{model_name} - {metric}', color=colors[model_idx * len(metrics_split_1) + metric_idx], linestyle='-', marker='o', markersize=4)
    
    plt.title('Index-wise Metrics for All Models (Precision, Recall, F1 Score, EF)', fontsize=18)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.05, 1))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    
    # Save the plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    img_bytes.seek(0)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    st.write("## Combined Metrics Analysis (Part 1)")
    st.markdown(
        f'<div style="overflow-x: auto; width: 100%;"><img src="data:image/png;base64,{img_base64}" style="width: 400%; height: auto;"></div>',
        unsafe_allow_html=True
    )
    
    # Download button for the first plot
    st.download_button(
        label="Download Plot 1",
        data=img_bytes.getvalue(),
        file_name="combined_metrics_part1.png",
        mime="image/png"
    )
    
    # Create the second figure
    plt.figure(figsize=(100, 20))
    colors = sns.color_palette("husl", n_colors=len(models) * len(metrics_split_2))
    
    for model_idx, model_name in enumerate(models):
        for metric_idx, metric in enumerate(metrics_split_2):
            metric_data = [data[str(i)][f"{model_name}_hallucination_metrics"][metric] for i in range(400)]
            plt.plot(metric_data, label=f'{model_name} - {metric}', color=colors[model_idx * len(metrics_split_2) + metric_idx], linestyle='-', marker='o', markersize=4)
    
    plt.title('Index-wise Metrics for All Models (PH, OF, NH, LF)', fontsize=18)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.05, 1))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    
    # Save the second plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    img_bytes.seek(0)
    
    # Encode the second image as base64
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    st.write("## Combined Metrics Analysis (Part 2)")
    st.markdown(
        f'<div style="overflow-x: auto; width: 100%;"><img src="data:image/png;base64,{img_base64}" style="width: 400%; height: auto;"></div>',
        unsafe_allow_html=True
    )
    
    # Download button for the second plot
    st.download_button(
        label="Download Plot 2",
        data=img_bytes.getvalue(),
        file_name="combined_metrics_part2.png",
        mime="image/png"
    )
    
    # Add analysis reports
    st.write("## Analysis for Part 1:")
    st.write("1. **Precision**: Higher Precision values indicate models that are more accurate in generating relevant information.")
    st.write("2. **Recall**: Higher Recall values suggest that the model captures a larger portion of the relevant information present in the input.")
    st.write("3. **F1 Score**: The F1 Score combines Precision and Recall; models with higher F1 Scores balance accuracy and coverage effectively.")
    st.write("4. **Extractiveness Factor (EF)**: Models with high EF values focus more on generating summaries closely aligned with the input text.")
    
    st.write("## Analysis for Part 2:")
    st.write("1. **Positive Hallucination (PH)**: High PH values indicate that the model might be generating additional, potentially irrelevant information.")
    st.write("2. **Over Focus Factor (OF)**: High OF values suggest that the model may be overly focused on specific aspects, potentially missing broader context.")
    st.write("3. **Negative Hallucination (NH)**: High NH values reflect the model's tendency to omit important information from the summaries.")
    st.write("4. **Lost Focus (LF)**: High LF values suggest that the model struggles to maintain focus on relevant content, which could impact summary accuracy.")
    
import io
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# Function to plot RHI metrics for all models index-wise
def plot_rhi_metrics_indexwise(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo", "RefSum"]
    rhi_metric = 'rhi'
    
    # Determine the data length based on the maximum index present in the data
    data_length = max(int(idx) for idx in data.keys()) + 1
    
    # Use seaborn color palette for distinct colors
    colors = sns.color_palette("husl", n_colors=len(models)).as_hex()
    
    # Create a Plotly figure for interactive plotting
    fig = go.Figure()
    
    models_with_data = []
    for model_idx, model_name in enumerate(models):
        # Extract RHI data for the current model
        rhi_data = [
            data.get(str(i), {}).get(f"{model_name}_hallucination_metrics", {}).get(rhi_metric, None)
            for i in range(data_length)
        ]
        
        # Filter out None values and track indices with data
        valid_rhi_data = [(i, val) for i, val in enumerate(rhi_data) if val is not None]
        
        if valid_rhi_data:
            indices, values = zip(*valid_rhi_data)
            # Add trace for each model with distinct colors
            fig.add_trace(go.Scatter(
                x=indices,
                y=values,
                mode='lines+markers',
                name=model_name,
                marker=dict(color=colors[model_idx], size=6),
                line=dict(color=colors[model_idx]),
                hovertemplate="Index: %{x}<br>RHI: %{y:.2f}"  # Show index and RHI value on hover
            ))
            models_with_data.append(model_name)
        else:
            print(f"No valid RHI data to plot for model: {model_name}")
    
    # Configure plot layout
    fig.update_layout(
        title="Index-wise Relation Hallucination Index (RHI) for All Models",
        xaxis_title="Index",
        yaxis_title="RHI Value",
        template="plotly_white",
        height=600,
        legend_title="Models",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05)
    )
    
    # Display the plot in Streamlit
    st.write("## RHI Metrics Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # Save the plot as a downloadable image
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format='png')
    img_bytes.seek(0)
    
    # Download button for the plot
    st.download_button(
        label="Download RHI Plot",
        data=img_bytes,
        file_name="rhi_metrics.png",
        mime="image/png"
    )

    # Example analysis section
    st.write("### Analysis:")
    st.write("1. **RHI**: The Relation Hallucination Index (RHI) helps assess the degree of hallucination in the generated summaries. Higher RHI values indicate a higher level of hallucination, meaning the model may be generating less accurate or more fabricated content. Evaluate how each model's RHI varies across different indices to identify trends, patterns, or anomalies in performance.")
    
    
def plot_rhi_per_model(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_metric = 'rhi'
    
    # Create a figure for each model
    for model_name in models:
        rhi_data = []
        for i in range(400):
            model_key = f"{model_name}_hallucination_metrics"
            model_data = data.get(str(i), {})
            if model_key in model_data:
                rhi_data.append(model_data[model_key].get(rhi_metric, None))
        
        # Filter out None values
        rhi_data = [value for value in rhi_data if value is not None]
        
        # Plot each model's RHI metric if there's data
        if rhi_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set(style="whitegrid")
            ax.plot(rhi_data, label=model_name, linestyle='-', marker='o', markersize=4)
            
            # Configure plot settings
            ax.set_title(f'Relation Hallucination Index (RHI) for {model_name}', fontsize=16)
            ax.set_xlabel('Index', fontsize=14)
            ax.set_ylabel('RHI Value', fontsize=14)
            ax.legend(loc='upper right', fontsize='small')
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(True)
            
            # Display the plot
            st.write(f"## RHI Metrics Analysis for {model_name}")
            st.pyplot(fig)
            
            # Save the plot to a BytesIO object for download
            img_bytes = io.BytesIO()
            fig.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
            img_bytes.seek(0)
            
            # Download button for the plot
            st.download_button(
                label=f"Download RHI Plot for {model_name}",
                data=img_bytes,
                file_name=f"rhi_metrics_{model_name}.png",
                mime="image/png"
            )
        else:
            st.write(f"No valid RHI data to plot for model: {model_name}")

    # Example analysis
    st.write("### Analysis:")
    st.write("1. **RHI**: The Relation Hallucination Index (RHI) for each model helps to assess the degree of hallucination separately. This allows you to compare the hallucination levels across different models more clearly. Observe how the RHI values for each model vary across different indices to understand the performance and identify any anomalies or patterns.")


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
def plot_rhi_indexwise(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    # Collect RHI values for each model
    for i in range(400):
        for model_name in models:
            # Use only the first occurrence of each model
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)  # Use NaN for missing values

    plt.figure(figsize=(14, 8))

    # Plot each model's RHI values
    for model_name in models:
        plt.plot(range(400), rhi_values_by_model[model_name], label=model_name)

    plt.title('RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
    
def plot_rhi_indexwise_separate(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        print(f"Processing index {i}...")
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)
            
            if isinstance(rhi_value, list):
                print(f"  Raw RHI for {model_name} at index {i}: {rhi_value}")
                # Aggregate list values into a single value (e.g., mean)
                rhi_value = np.mean(rhi_value)
                print(f"  Aggregated RHI for {model_name} at index {i}: {rhi_value}")
            
            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)  # Use NaN for missing values

    for model_name in models:
        print(f"RHI Values for {model_name}: {rhi_values_by_model[model_name]}")
        print(f"Plotting RHI for {model_name}...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(400), rhi_values_by_model[model_name], label=model_name)
        plt.title(f'RHI Index-wise for {model_name}')
        plt.xlabel('Index')
        plt.ylabel('RHI')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)  # Display the plot in Streamlit
        plt.close()
        print(f"Plot for {model_name} generated.")
    
def plot_combined_cdf_vs_rhi(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    import io
    import streamlit as st
    
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_metric = 'rhi'
    
    # Color palette for distinct colors
    colors = sns.color_palette("husl", n_colors=len(models))
    
    # Initialize dictionaries to store the best and worst performances
    best_models = {}
    worst_models = {}
    
    # Part 1: Combined Plot for All Models
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    sns.set(style="whitegrid")
    
    all_rhi_data = {}
    
    for idx, model_name in enumerate(models):
        # Extract RHI data for the current model
        rhi_data = []
        for i in range(400):
            model_key = f"{model_name}_hallucination_metrics"
            model_data = data.get(str(i), {})
            if model_key in model_data:
                rhi_data.append(model_data[model_key].get(rhi_metric, None))
        
        # Filter out None values
        rhi_data = [value for value in rhi_data if value is not None]
        
        # Sort RHI data for CDF calculation
        sorted_rhi_data = np.sort(rhi_data)
        cdf = np.arange(len(sorted_rhi_data)) / float(len(sorted_rhi_data))
        
        all_rhi_data[model_name] = (sorted_rhi_data, cdf)
        
        # Plot CDF vs RHI for the model
        if sorted_rhi_data.size > 0:
            ax_combined.plot(sorted_rhi_data, cdf, label=model_name, color=colors[idx], linestyle='-', marker='o', markersize=4)
    
    # Configure the combined plot settings
    ax_combined.set_title('CDF vs RHI for All Models Combined', fontsize=18)
    ax_combined.set_xlabel('RHI Value', fontsize=14)
    ax_combined.set_ylabel('Cumulative Probability', fontsize=14)
    ax_combined.legend(loc='lower right', fontsize='small')
    ax_combined.tick_params(axis='x', labelsize=10)
    ax_combined.tick_params(axis='y', labelsize=10)
    ax_combined.grid(True)
    
    # Display the combined plot
    st.write("## Combined CDF vs RHI Metrics Analysis for All Models")
    st.pyplot(fig_combined)
    
    # Save the combined plot to a BytesIO object for download
    img_bytes_combined = io.BytesIO()
    fig_combined.savefig(img_bytes_combined, format='png', bbox_inches='tight', pad_inches=0.1)
    img_bytes_combined.seek(0)
    
    # Download button for the combined plot
    st.download_button(
        label="Download Combined CDF vs RHI Plot for All Models",
        data=img_bytes_combined,
        file_name="cdf_vs_rhi_metrics_combined.png",
        mime="image/png"
    )
    
    # Combined analysis report
    st.write("### Combined Analysis Report:")
    st.write("1. **All Models**: The combined CDF plot shows how each model performs in relation to hallucination tendencies. Models with steeper curves at lower RHI values generally produce more reliable summaries, while more gradual curves indicate higher hallucination tendencies.")
    st.write("2. **Model Comparison**: Differences in CDF curves between models in the combined plot can highlight areas where one model may outperform another in terms of maintaining factual accuracy.")
    
    # Part 2: Pairwise Plots for Models
    model_pairs = list(itertools.combinations(models, 2))
    for pair_idx, (model_1, model_2) in enumerate(model_pairs):
        # Prepare figure for the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set(style="whitegrid")
        
        for idx, model_name in enumerate([model_1, model_2]):
            sorted_rhi_data, cdf = all_rhi_data[model_name]
            
            # Plot CDF vs RHI for the model
            if sorted_rhi_data.size > 0:
                ax.plot(sorted_rhi_data, cdf, label=model_name, color=colors[idx], linestyle='-', marker='o', markersize=4)
        
        # Configure the pairwise plot settings
        ax.set_title(f'CDF vs RHI for {model_1} and {model_2}', fontsize=18)
        ax.set_xlabel('RHI Value', fontsize=14)
        ax.set_ylabel('Cumulative Probability', fontsize=14)
        ax.legend(loc='lower right', fontsize='small')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True)
        
        # Display the pairwise plot
        st.write(f"## CDF vs RHI Metrics Analysis for {model_1} and {model_2}")
        st.pyplot(fig)
        
        # Save the pairwise plot to a BytesIO object for download
        img_bytes_pair = io.BytesIO()
        fig.savefig(img_bytes_pair, format='png', bbox_inches='tight', pad_inches=0.1)
        img_bytes_pair.seek(0)
        
        # Download button for the pairwise plot
        st.download_button(
            label=f"Download CDF vs RHI Plot for {model_1} and {model_2}",
            data=img_bytes_pair,
            file_name=f"cdf_vs_rhi_metrics_{model_1}_vs_{model_2}.png",
            mime="image/png"
        )
        
        # Pairwise analysis report
        st.write("### Pairwise Analysis Report:")
        st.write(f"1. **{model_1} vs {model_2}**: The comparison between `{model_1}` and `{model_2}` highlights how each model manages hallucination across different indices. A steeper CDF curve at lower RHI values suggests more accurate summaries, while a more gradual curve indicates a higher tendency towards hallucination.")
        st.write(f"2. **Trends and Patterns**: Assess how the CDF curves for `{model_1}` and `{model_2}` progress. Significant differences in the curves may suggest one model's superiority in certain scenarios, or highlight specific indices where one model underperforms relative to the other.")
        st.write(f"3. **Key Observations**: Look for any anomalies, such as sharp jumps in the curves, which might indicate inconsistent model behavior at specific indices.")
        
        # Determine best and worst performing models
        if np.max(cdf) < 0.5:
            best_models[model_1] = sorted_rhi_data
            worst_models[model_2] = sorted_rhi_data
        elif np.max(cdf) > 0.5:
            best_models[model_2] = sorted_rhi_data
            worst_models[model_1] = sorted_rhi_data
    
        # Final summary of model performances
        if best_models:
            best_model = min(best_models, key=lambda x: np.mean(best_models[x]))
            st.write(f"**Best Performing Model:** {best_model} - This model consistently has lower RHI values, indicating a tendency to produce more accurate summaries with less hallucination.")
        else:
            st.write("No best performing model could be determined due to lack of data.")

        if worst_models:
            worst_model = max(worst_models, key=lambda x: np.mean(worst_models[x]))
            st.write(f"**Worst Performing Model:** {worst_model} - This model shows higher RHI values, indicating a tendency towards higher hallucination in generated summaries.")
        else:
            st.write("No worst performing model could be determined due to lack of data.")


    import streamlit as st
import json
import pandas as pd
import io
from json import *


def save_json_to_string(data, file_name=None):
    json_str = json.dumps(data, indent=4)
    if file_name:
        with open(file_name, 'w') as f:
            f.write(json_str)
    return json_str

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_rhi_performance(data):
    # Define the range for good performance
    lower_threshold = 0.1
    upper_threshold = 1.9

    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}
    good_performance_indices = {model: [] for model in models}
    bad_performance_indices = {model: [] for model in models}
    good_counts = {model: 0 for model in models}
    bad_counts = {model: 0 for model in models}

    # Collect RHI values for each model and categorize based on performance
    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
                
                # Categorize based on the defined range
                if lower_threshold <= rhi_value <= upper_threshold:
                    good_performance_indices[model_name].append(i)
                    good_counts[model_name] += 1
                else:
                    bad_performance_indices[model_name].append(i)
                    bad_counts[model_name] += 1
            else:
                rhi_values_by_model[model_name].append(np.nan)  # Use NaN for missing values

    # Plotting RHI values where models performed well
    plt.figure(figsize=(14, 8))
    for model_name in models:
        plt.plot(good_performance_indices[model_name], 
                 [rhi_values_by_model[model_name][i] for i in good_performance_indices[model_name]], 
                 label=model_name)

    plt.title('RHI Index-wise for Each Model (Good Performance - RHI in Range)')
    plt.xlabel('Index')
    plt.ylabel('RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # Plotting RHI values where models performed poorly
    plt.figure(figsize=(14, 8))
    for model_name in models:
        plt.plot(bad_performance_indices[model_name], 
                 [rhi_values_by_model[model_name][i] for i in bad_performance_indices[model_name]], 
                 label=model_name)

    plt.title('RHI Index-wise for Each Model (Bad Performance - RHI Out of Range)')
    plt.xlabel('Index')
    plt.ylabel('RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # Plotting counts of good and bad performance
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    indices = np.arange(len(models))

    good_values = [good_counts[model] for model in models]
    bad_values = [bad_counts[model] for model in models]

    plt.bar(indices, good_values, bar_width, label='Good Performance (RHI in Range)', color='green')
    plt.bar(indices + bar_width, bad_values, bar_width, label='Bad Performance (RHI Out of Range)', color='red')

    plt.xlabel('Models')
    plt.ylabel('Number of Indices')
    plt.title('Number of Good and Bad Performance Indices for Each Model')
    plt.xticks(indices + bar_width / 2, models)
    plt.legend()
    st.pyplot(plt)

    # Normalizing RHI values
    normalized_rhi_values_by_model = {}
    for model_name in models:
        rhi_values = np.array(rhi_values_by_model[model_name])
        min_rhi = np.nanmin(rhi_values)
        max_rhi = np.nanmax(rhi_values)
        normalized_rhi = (rhi_values - min_rhi) / (max_rhi - min_rhi)
        normalized_rhi_values_by_model[model_name] = normalized_rhi

    # Plotting normalized RHI values
    plt.figure(figsize=(14, 8))
    for model_name in models:
        plt.plot(range(400), normalized_rhi_values_by_model[model_name], label=model_name)

    plt.title('Normalized RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Normalized RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

  
# Normalization:
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def min_max_normalize(data):
    min_value = np.nanmin(data)
    max_value = np.nanmax(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

def plot_rhi_indexwise_min_max(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)

    plt.figure(figsize=(14, 8))

    # Apply Min-Max Normalization and plot each model's RHI values
    for model_name in models:
        normalized_rhi = min_max_normalize(np.array(rhi_values_by_model[model_name]))
        plt.plot(range(400), normalized_rhi, label=model_name)

    plt.title('Min-Max Normalized RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Normalized RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def z_score_normalize(data):
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def plot_rhi_indexwise_z_score(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)

    plt.figure(figsize=(14, 8))

    # Apply Z-Score Normalization and plot each model's RHI values
    for model_name in models:
        normalized_rhi = z_score_normalize(np.array(rhi_values_by_model[model_name]))
        plt.plot(range(400), normalized_rhi, label=model_name)

    plt.title('Z-Score Normalized RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Normalized RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def robust_normalize(data):
    median = np.nanmedian(data)
    iqr = np.nanpercentile(data, 75) - np.nanpercentile(data, 25)
    normalized_data = (data - median) / iqr
    return normalized_data

def plot_rhi_indexwise_robust(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)

    plt.figure(figsize=(14, 8))

    # Apply Robust Normalization and plot each model's RHI values
    for model_name in models:
        normalized_rhi = robust_normalize(np.array(rhi_values_by_model[model_name]))
        plt.plot(range(400), normalized_rhi, label=model_name)

    plt.title('Robust Normalized RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Normalized RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
def l1_normalize(data):
    l1_norm = np.sum(np.abs(data), axis=0)
    normalized_data = data / l1_norm
    return normalized_data

def plot_rhi_indexwise_l1(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)

    plt.figure(figsize=(14, 8))

    # Apply L1 Normalization and plot each model's RHI values
    for model_name in models:
        normalized_rhi = l1_normalize(np.array(rhi_values_by_model[model_name]))
        plt.plot(range(400), normalized_rhi, label=model_name)

    plt.title('L1 Normalized RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Normalized RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
def l2_normalize(data):
    l2_norm = np.sqrt(np.sum(np.square(data), axis=0))
    normalized_data = data / l2_norm
    return normalized_data

def plot_rhi_indexwise_l2(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)

    plt.figure(figsize=(14, 8))

    # Apply L2 Normalization and plot each model's RHI values
    for model_name in models:
        normalized_rhi = l2_normalize(np.array(rhi_values_by_model[model_name]))
        plt.plot(range(400), normalized_rhi, label=model_name)

    plt.title('L2 Normalized RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Normalized RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)


def log_transform(data):
    data = np.array(data)
    data = np.where(data > 0, data, np.nan)  # Avoid log(0) and negative values
    transformed_data = np.log1p(data)  # log1p(x) = log(x + 1)
    return transformed_data

def plot_rhi_indexwise_log(data):
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo","RefSum"]
    rhi_values_by_model = {model: [] for model in models}

    for i in range(400):
        for model_name in models:
            model_key = f"{model_name}_hallucination_metrics"
            rhi_value = data.get(str(i), {}).get(model_key, {}).get('rhi', None)

            if isinstance(rhi_value, list):
                rhi_value = np.mean(rhi_value)  # Aggregate list values

            if rhi_value is not None:
                rhi_values_by_model[model_name].append(rhi_value)
            else:
                rhi_values_by_model[model_name].append(np.nan)

    plt.figure(figsize=(14, 8))

    # Apply Log Transformation and plot each model's RHI values
    for model_name in models:
        transformed_rhi = log_transform(np.array(rhi_values_by_model[model_name]))
        plt.plot(range(400), transformed_rhi, label=model_name)

    plt.title('Log Transformed RHI Index-wise for Each Model')
    plt.xlabel('Index')
    plt.ylabel('Transformed RHI')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Function to plot various metrics for all models index-wise
def plot_all_metrics_indexwise(data):
    metrics = ['ef', 'ph', 'of', 'nh', 'lf', 'rhi']
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo", "RefSum"]
    
    # Use seaborn color palette for distinct colors
    colors = sns.color_palette("husl", n_colors=len(models)).as_hex()
    
    # Loop over each metric and create a separate plot for each one
    for metric in metrics:
        fig = go.Figure()
        models_with_data = []
        
        for model_idx, model_name in enumerate(models):
            # Extract metric data for the current model
            metric_data = [
                data.get(str(i), {}).get(f"{model_name}_hallucination_metrics", {}).get(metric, None)
                for i in range(max(int(idx) for idx in data.keys()) + 1)
            ]
            
            # Filter out None values and track indices with data
            valid_metric_data = [(i, val) for i, val in enumerate(metric_data) if val is not None]
            
            if valid_metric_data:
                indices, values = zip(*valid_metric_data)
                # Add trace for each model with distinct colors
                fig.add_trace(go.Scatter(
                    x=indices,
                    y=values,
                    mode='lines+markers',
                    name=model_name,
                    marker=dict(color=colors[model_idx], size=6),
                    line=dict(color=colors[model_idx]),
                    hovertemplate=f"Index: {{%x}}<br>{metric.capitalize()}: {{%y:.2f}}"  # Show index and metric value on hover
                ))
                models_with_data.append(model_name)
            else:
                print(f"No valid data to plot for model: {model_name} for metric: {metric}")
        
        # Configure plot layout for each metric
        fig.update_layout(
            title=f"Index-wise {metric.capitalize()} for All Models",
            xaxis_title="Index",
            yaxis_title=f"{metric.capitalize()} Value",
            template="plotly_white",
            height=600,
            legend_title="Models",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05)
        )
        
        # Display each plot in Streamlit
        st.write(f"## {metric.capitalize()} Metric Analysis")
        st.plotly_chart(fig, use_container_width=True)

        # Save each plot as a downloadable image
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format='png')
        img_bytes.seek(0)
        
        # Download button for each plot
        st.download_button(
            label=f"Download {metric.capitalize()} Plot",
            data=img_bytes,
            file_name=f"{metric}_metrics.png",
            mime="image/png"
        )

        # Example analysis section
        st.write("### Analysis:")
        st.write(f"1. **{metric.capitalize()}**: This metric provides insight into the model's performance for {metric}. Track trends across indices to observe how the model's {metric} score varies.")

import pandas as pd

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io

# Function to calculate, display, and download average metrics table with enhanced styling
def display_average_metrics_table(data):
    metrics = ['ef', 'ph', 'of', 'nh', 'lf', 'Lost Hallucination', 'rhi']
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo", "RefSum"]
    
    # Initialize dictionary to store average metrics for each model
    avg_metrics = {model: {metric: 0 for metric in metrics} for model in models}
    model_counts = {model: {metric: 0 for metric in metrics} for model in models}

    # Loop through data to sum up the values for each metric and count valid entries
    for idx in data.keys():
        for model in models:
            for metric in metrics:
                value = data.get(idx, {}).get(f"{model}_hallucination_metrics", {}).get(metric, None)
                if value is not None:
                    avg_metrics[model][metric] += value
                    model_counts[model][metric] += 1

    # Calculate average by dividing summed values by counts
    for model in models:
        for metric in metrics:
            if model_counts[model][metric] > 0:
                avg_metrics[model][metric] /= model_counts[model][metric]

    # Convert dictionary to DataFrame for display
    avg_df = pd.DataFrame(avg_metrics).T
    avg_df.columns = [metric.upper() for metric in metrics]  # Capitalize metric names for display

    # Display table in Streamlit
    st.write("### Average Metrics Across All Models")
    st.table(avg_df)
    
    # Generate an image of the table with styled borders for download
    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(avg_df)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=avg_df.values,
        colLabels=avg_df.columns,
        rowLabels=avg_df.index,
        cellLoc='center',
        loc='center',
    )

    # Style the table with borders and font adjustments
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(avg_df.columns))))

    # Set cell border color and width dynamically based on the table size
    for (i, j), cell in table._cells.items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)
        if i == 0 or j == -1:  # Header and row labels
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4F81BD")  # Header background color
        else:
            cell.set_facecolor("#F2F2F2")  # Light gray for data cells

    # Save table as image in memory
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches="tight", dpi=300)
    img_bytes.seek(0)

    # Download button for the table image
    st.download_button(
        label="Download Table as Image",
        data=img_bytes,
        file_name="average_metrics_table.png",
        mime="image/png"
    )

import pandas as pd
import streamlit as st
import io

import matplotlib.pyplot as plt
import pandas as pd
import io
import streamlit as st
import tempfile  # Import tempfile for creating temporary files

def display_average_metrics_latex_table(data):
    metrics = ['ef', 'ph', 'of', 'nh', 'lf', 'Lost Hallucination', 'rhi']
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo", "RefSum"]

    # Initialize dictionary to store average metrics for each model
    avg_metrics = {model: {metric: 0 for metric in metrics} for model in models}
    model_counts = {model: {metric: 0 for metric in metrics} for model in models}

    # Loop through data to sum up the values for each metric and count valid entries
    for idx in data.keys():
        for model in models:
            for metric in metrics:
                value = data.get(idx, {}).get(f"{model}_hallucination_metrics", {}).get(metric, None)
                if value is not None:
                    avg_metrics[model][metric] += value
                    model_counts[model][metric] += 1

    # Calculate average by dividing summed values by counts
    for model in models:
        for metric in metrics:
            if model_counts[model][metric] > 0:
                avg_metrics[model][metric] /= model_counts[model][metric]

    # Convert dictionary to DataFrame for processing
    avg_df = pd.DataFrame(avg_metrics).T
    avg_df.columns = [metric.upper() for metric in metrics]  # Capitalize metric names for display

    # Generate LaTeX table with enhanced style
    latex_code = avg_df.to_latex(
        bold_rows=True,
        index=True,
        caption="\\textbf{Average Metrics Across All Models}",
        label="tab:average_metrics",
        column_format="|l|" + "c|" * len(avg_df.columns),  # Add vertical lines for table style
        escape=False,
        header=True
    )

    # Display table in Streamlit
    st.write("### Average Metrics Across All Models")
    st.write(avg_df)

    # Download button for LaTeX code
    latex_bytes = io.BytesIO(latex_code.encode('utf-8'))
    st.download_button(
        label="Download LaTeX Table",
        data=latex_bytes,
        file_name="average_metrics_table.tex",
        mime="text/plain"
    )
    
    # Create a plot of the table using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    
    # Create table and add it to the plot
    table = ax.table(cellText=avg_df.values,
                    colLabels=avg_df.columns,
                    rowLabels=avg_df.index,
                    loc='center',
                    cellLoc='center',
                    colColours=['#f1f1f1'] * len(avg_df.columns))  # Optional: color for column headers
    
    # Use tempfile to create a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        image_path = tmpfile.name
        
    # Save the table as an image
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.05)

    # Display the image in Streamlit
    st.image(image_path)

    # Provide a download button for the image
    with open(image_path, "rb") as img_file:
        st.download_button(
            label="Download Image of Table",
            data=img_file,
            file_name="average_metrics_table.png",
            mime="image/png"
        )


# Example usage
# Assuming `data` is your dictionary containing the metrics data
# display_average_metrics_table(data)

# Example usage
# Assuming `data` is your dictionary containing the metrics data
# display_average_metrics_table(data)


import streamlit as st
import pandas as pd
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import io

def compute_and_plot_hallucination_stats(data):
    metrics = ['ef', 'ph', 'of', 'nh', 'lf', 'Lost Hallucination', 'rhi']
    models = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo", "RefSum"]

    # Initialize data storage for statistics
    stats = {metric: {model: [] for model in models} for metric in metrics}

    for idx in data.keys():
        for model in models:
            for metric in metrics:
                value = data.get(idx, {}).get(f"{model}_hallucination_metrics", {}).get(metric, None)
                if value is not None:
                    stats[metric][model].append(value)

    # Compute mean and variance for each metric per model
    stats_df = {metric: {} for metric in metrics}
    for metric in metrics:
        for model in models:
            values = stats[metric][model]
            if values:
                stats_df[metric][f"{model}_mean"] = np.mean(values)
                stats_df[metric][f"{model}_variance"] = np.var(values)
            else:
                stats_df[metric][f"{model}_mean"] = 0
                stats_df[metric][f"{model}_variance"] = 0

    # Convert dictionary to DataFrame
    stats_df = pd.DataFrame(stats_df).T
    st.write("### Hallucination Metrics Statistics")
    st.table(stats_df)

    # Generate bar plots for each model
    for model in models:
        model_means = {metric: stats_df.loc[metric, f"{model}_mean"] for metric in metrics}
        fig = px.bar(
            x=list(model_means.keys()), 
            y=list(model_means.values()), 
            labels={'x': 'Metrics', 'y': 'Mean'},
            title=f"{model} - Hallucination Metrics",
            color=list(model_means.keys())
        )
        st.plotly_chart(fig, use_container_width=True)

        # Allow downloading of individual plots
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png")
        st.download_button(
            label=f"Download {model} Plot",
            data=buffer.getvalue(),
            file_name=f"{model}_hallucination_plot.png",
            mime="image/png"
        )

    # Generate combined plot for RHI only
    rhi_data = []
    for model in models:
        rhi_data.append({
            "Model": model,
            "Mean": stats_df.loc["rhi", f"{model}_mean"],
            "Variance": stats_df.loc["rhi", f"{model}_variance"]
        })

    rhi_df = pd.DataFrame(rhi_data)

    # Plot Mean and Variance for RHI across models
    fig = px.bar(
        rhi_df, 
        x="Model", 
        y=["Mean", "Variance"], 
        barmode="group",
        title="Mean and Variance of RHI Across Models"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Allow downloading of the RHI plot
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    st.download_button(
        label="Download RHI Mean & Variance Plot",
        data=buffer.getvalue(),
        file_name="rhi_hallucination_plot.png",
        mime="image/png"
    )

def main():
    st.title("🔍 Relation Hallucination and Extraction Analysis")

    # Upload JSON File
    uploaded_file = st.file_uploader("📂 Upload JSON File", type="json")
    if uploaded_file:
        try:
            json_data = json.load(uploaded_file)
            st.session_state.data = json_data

            # Option to process and extract relations
            process_and_extract = st.checkbox("⚙️ Process and Extract SVO Relations")
            if process_and_extract:
                with st.spinner("🔄 Processing the dataset..."):
                    st.session_state.processed_data = process_json_and_extract_relations(json_data)
                st.success("✅ Processing complete!")

            # Check if 'processed_data' exists in session_state
            if 'processed_data' in st.session_state:
                df = pd.json_normalize(st.session_state.processed_data).T  # Transpose for better display
                df.index = df.index.map(str)  # Convert index to string

                # Convert nested dictionaries or lists to strings for better display in table
                for column in df.columns:
                    df[column] = df[column].apply(
                        lambda x: json.dumps(x, indent=2) if isinstance(x, (dict, list)) else x
                    )

                # Display the entire DataFrame as a table
                st.dataframe(df)

                # Save and Download processed data
                json_str = save_json_to_string(st.session_state.processed_data)
                st.download_button(
                    label="💾 Download Processed Data",
                    data=json_str,
                    file_name='processed_data.json',
                    mime='application/json'
                )
            else:
                st.warning("⚠️ Relation extraction has to be completed.")
                
        except ValueError as e:
            st.error("❌ Invalid JSON file. Please upload a valid JSON file.")

    # Dropdown menu for selecting the type of plot or analysis
    if 'processed_data' in st.session_state:
        option = st.selectbox(
            "📊 Select Analysis Type",
            [
                "---Select an option---",
                "📈 ROUGE Scores",
                "📉 Hallucination Metrics",
                "📑 Metrics Statistics",
                "📑 Hallucination Report",
                "📊 CDF and RHI Graph and Analysis",
                "📊 RHI Metric Graph",
                "📊 PLOT all metrics indexwise:",
                "📑 Average Metrics Table",
                "🧮 Compute Hallucination Metrics",
                "📈 CDF of Metrics",
                "📉 Correlation Analysis Report and Visualization",
                "📉 Correlation Analysis Line Plots",
                "📊 Statistical Analysis",
                "📈 Combined ROUGE Scores and Hallucination Metrics",
                "📊 RHI Plot for all Model",
                "📊 Plot RHI Index-wise (Separate)",
                "📉 Plot RHI Performance",
                "Display Table for all metrics",
                "📊 Min-Max Normalized RHI",
                "📉 Z-Score Normalized RHI",
                "📈 Robust Normalized RHI",
                "📉 L1 Normalized RHI",
                "📈 L2 Normalized RHI",
                "🔍 Log Transformed RHI"
            ]
        )

        # Execute the corresponding function based on the selection
        if option != "---Select an option---":
            if option == "📈 ROUGE Scores":
                st.subheader("📈 ROUGE Scores")
                plot_rouge_scores(st.session_state.processed_data)

            elif option == "📉 Hallucination Metrics":
                st.subheader("📉 Hallucination Metrics")
                plot_hallucination_metrics(st.session_state.processed_data)

            elif option == "📑 Metrics Statistics" :
                st.subheader("📑 Metrics Statistics")
                compute_and_plot_hallucination_stats(st.session_state.processed_data)
            elif option == "📑 Hallucination Report":
                st.subheader("📑 Hallucination Report")
                plot_hallucination_metrics_indexwise(st.session_state.processed_data)

            elif option == "📈 Combined Plot of Hallucination Metrics":
                st.subheader("📈 Combined Plot of Hallucination Metrics")
                plot_all_metrics_indexwise(st.session_state.processed_data)
                
            elif option == "📑 Average Metrics Table":
                st.subheader("📑 Average Metrics Table")
                display_average_metrics_table(st.session_state.processed_data)

            elif option == "📊 RHI Metric Graph":
                st.subheader("📊 RHI Metric Graph")
                plot_rhi_metrics_indexwise(st.session_state.processed_data)
            
            elif option == "📊 PLOT all metrics indexwise:":
                st.subheader("📊 PLOT all metrics indexwise:")
                plot_all_metrics_indexwise(st.session_state.processed_data)

            elif option == "📈 CDF of Metrics":
                st.subheader("📈 Cumulative Distribution Function (CDF) of Metrics")
                plot_cdf_graphs(st.session_state.processed_data)

            elif option == "📉 Correlation Analysis Report and Visualization":
                st.subheader("📉 Correlation Analysis Report and Visualization")
                plot_metrics_with_analysis(st.session_state.processed_data)

            elif option == "📉 Correlation Analysis Line Plots":
                st.subheader("📉 Correlation Analysis Line Plots")
                plot_correlation_analysis_lineplot(st.session_state.processed_data)

            elif option == "📊 Statistical Analysis":
                st.subheader("📊 Statistical Analysis")
                plot_statistical_analysis(st.session_state.processed_data)

            elif option == "📈 Combined ROUGE Scores and Hallucination Metrics":
                st.subheader("📈 Combined ROUGE Scores and Hallucination Metrics")
                plot_combined_rouge_and_metrics(st.session_state.processed_data)

            elif option == "📊 CDF and RHI Graph and Analysis":
                st.subheader("📊 CDF and RHI Graph and Analysis")
                plot_combined_cdf_vs_rhi(st.session_state.processed_data)

            elif option == "📊 RHI Plot for all Model":
                st.subheader("📊 RHI Plot for Each Model")
                plot_rhi_indexwise(st.session_state.processed_data)

            elif option == "📊 Plot RHI Index-wise (Separate)":
                st.subheader("📊 Plot RHI Index-wise (Separate)")
                plot_rhi_indexwise_separate(st.session_state.processed_data)

            elif option == "📉 Plot RHI Performance":
                st.subheader("📉 Plot RHI Performance")
                plot_rhi_performance(st.session_state.processed_data)

            elif option == "📊 Min-Max Normalized RHI":
                st.subheader("📊 Min-Max Normalized RHI")
                plot_rhi_indexwise_min_max(st.session_state.processed_data)

            elif option == "📉 Z-Score Normalized RHI":
                st.subheader("📉 Z-Score Normalized RHI")
                plot_rhi_indexwise_z_score(st.session_state.processed_data)

            elif option == "📈 Robust Normalized RHI":
                st.subheader("📈 Robust Normalized RHI")
                plot_rhi_indexwise_robust(st.session_state.processed_data)

            elif option == "📉 L1 Normalized RHI":
                st.subheader("📉 L1 Normalized RHI")
                plot_rhi_indexwise_l1(st.session_state.processed_data)

            elif option == "📈 L2 Normalized RHI":
                st.subheader("📈 L2 Normalized RHI")
                plot_rhi_indexwise_l2(st.session_state.processed_data)

            elif option == "🔍 Log Transformed RHI":
                st.subheader("🔍 Log Transformed RHI")
                plot_rhi_indexwise_log(st.session_state.processed_data)
            elif option == "Display Table for all metrics":
                st.subheader("Display Table for all metrics ")
                display_average_metrics_latex_table(st.session_state.processed_data)

if __name__ == "__main__":
    main()
