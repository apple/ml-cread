INPUT_MAX_LENGTH = 100
OUTPUT_MAX_LENGTH = 30  # Maximum sentence length to consider for output
MIN_COUNT = 5  # Minimum word count threshold for trimming


# Default word tokens
PAD_token = "[PAD]"  # Used for padding short sentences
SOS_token = "[SOS]"  # Start-of-sentence token
EOS_token = "[EOS]"  # End-of-sentence token
SOQ_token = "[SOQ]"  # start of context token
SOR_token = "[SOR]"
OOV_token = "[OOV]"  # OOV token
MASK_token = "[MASK]"

# A list of tokens with no meanings that should be removed in detokenization
indicator_tokens = [PAD_token, SOS_token, EOS_token, SOQ_token, SOR_token, MASK_token]
