def calculate_complexity_scores(game_id):
    """
    need to implement game complexity based on environment and rules
    """
    rules_complexity = calculate_rules_complexity(game_id)
    pass


def calculate_rules_complexity(game_id):
    # This is a simplistic manual mapping, and you should adjust it based on your game analysis
    complexity_mapping = {
        "ALE/Pong-v5": 1,
        "ALE/MontezumasRevenge-v5": 5,
        "ALE/MsPacman-v5": 3,
        "ALE/Hero-v5": 4,
        # Add more games as needed
    }
    return complexity_mapping.get(game_id, 1)  # Default to 1 if game not listed
