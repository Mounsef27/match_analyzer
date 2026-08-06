from pprint import pprint  # just a bib for pretty printing  :)

from transformers import pipeline

from match_analyzer.config.constants import PROMPT
from match_analyzer.config.models_dict import models_dict


def main():
    generator = pipeline(
        "text-generation",
        model=models_dict["google_models"]["gemma-2b"],
        device=-1,  # 0 : gpu 0, -1 : cpu
    )

    result = generator(PROMPT, max_new_tokens=20)

    pprint(result)  # just i want to see how it is formmated !!!!!!!!!!!!!


if __name__ == "__main__":
    main()
