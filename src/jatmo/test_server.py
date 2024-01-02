from prompt_injection_defense.server import rate_completions


def test_rate_completions():
    """
    Test function for rate_completions function.
    """
    prompts = [
        "What is the capital of France?",
        "What is the meaning of life?",
        "Who invented telephone?",
        "What is water made of?",
        "What language is spoken in Brazil?",
        "What is the capital of Australia?",
        "What is the largest mammal?",
        "What is the distance from Earth to Moon?",
        "Who wrote Romeo and Juliet?",
        "Where is the Eiffel Tower located?",
        "Who was the first man on the moon?",
        "What is the square root of 144?",
        "Who discovered gravity?",
        "What is the capital of Switzerland?",
        "Who painted the Mona Lisa?",
        "Who is the current president of the United States?",
        "What is the currency of Japan?",
        "What is the second planet in our solar system?",
        "Who wrote the Odyssey?",
        "What is the largest ocean?",
        "What is the freezing point of water in Celsius?",
        "What is the chemical symbol for Iron?",
        "Who wrote the Harry Potter series?",
        "Who invented the light bulb?",
        "What is the capital of Canada?",
    ]
    responses = [
        "Paris",
        "42",
        "Alexander Graham Bell",
        "Hydrogen and Oxygen",
        "Portuguese",
        "Canberra",
        "Blue Whale",
        "384,400 km",
        "William Shakespeare",
        "Paris",
        "Neil Armstrong",
        "12",
        "Isaac Newton",
        "Bern",
        "Leonardo da Vinci",
        "Joe Biden",
        "Yen",
        "Venus",
        "Homer",
        "Pacific Ocean",
        "0 degree",
        "Fe",
        "J.K. Rowling",
        "Thomas Edison",
        "Ottawa",
    ]

    rate_completions(prompts, responses, number_of_processes=16)


def main():
    """
    Run test functions.
    """
    test_rate_completions()


if __name__ == "__main__":
    main()
