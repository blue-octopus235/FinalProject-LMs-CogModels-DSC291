"""Verb-agreement flip. Copied verbatim from Kelly's noise-injection notebook
so that minimal-pair generation is CONSISTENT with how the training corpora
were corrupted. Do not diverge this from her version.

Requires: pip install lemminflect
"""
import string

from lemminflect import getLemma, getInflection


def flip_verb_agreement_perfect(verb, verb_pos):
    prefix = ""
    suffix = ""
    v_strip = verb
    while len(v_strip) > 0 and v_strip[0] in string.punctuation:
        prefix += v_strip[0]
        v_strip = v_strip[1:]
    while len(v_strip) > 0 and v_strip[-1] in string.punctuation:
        suffix = v_strip[-1] + suffix
        v_strip = v_strip[:-1]
    if not v_strip:
        return verb
    v_lower = v_strip.lower()
    if v_lower == 'vbz':
        return prefix + ('VBP' if v_strip.isupper() else 'vbp') + suffix
    if v_lower == 'vbp':
        return prefix + ('VBZ' if v_strip.isupper() else 'vbz') + suffix
    hard_s_to_p = {'is': 'are', 'was': 'were', 'has': 'have', 'does': 'do', 'goes': 'go'}
    hard_p_to_s = {'are': 'is', 'were': 'was', 'have': 'has', 'do': 'does', 'go': 'goes', 'am': 'is'}
    if verb_pos == 'VBZ' and v_lower in hard_s_to_p:
        flipped = hard_s_to_p[v_lower]
    elif v_lower in hard_p_to_s:
        flipped = hard_p_to_s[v_lower]
    else:
        target_pos = 'VBP' if verb_pos == 'VBZ' else 'VBZ'
        lemmas = getLemma(v_lower, upos='VERB')
        if lemmas:
            lemma = lemmas[0]
            inflections = getInflection(lemma, tag=target_pos)
            if inflections:
                flipped = inflections[0]
            else:
                flipped = v_lower + 's' if target_pos == 'VBZ' else (v_lower[:-1] if v_lower.endswith('s') else v_lower)
        else:
            flipped = v_lower + 's' if target_pos == 'VBZ' else (v_lower[:-1] if v_lower.endswith('s') else v_lower)
    if v_strip.istitle():
        flipped = flipped.capitalize()
    elif v_strip.isupper():
        flipped = flipped.upper()
    return prefix + flipped + suffix
