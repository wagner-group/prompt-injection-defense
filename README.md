# Jatmo
Fine-tuning base models to build robust task-specific models

## Installation

* Create a pyton3.9 virtual env: `python3.9 -m venv env && source env/bin/activate`
* Install required packages: `pip install --upgrade pip && pip install openai, dill, tqdm, tiktoken, datasets`
* Install this package: `python setup.py install`
* Export your openai key: `export OPENAI_API_KEY=[your key]`

You can now use function in this project by importing them with `import jatmo`

## Usage

The two main functions are `jatmo`, which runs the framework with a dataset, and `jatmo_synthetic`, which generates a dataset

### `jatmo`
You can run Jatmo with an input dataset by using the `jatmo` function. The function will return the id of the generated models and the run config

```
from jatmo import jatmo

### Load inputs into array
model_ids, config = jatmo( inputs, task="Determine whether the following comment is positive or negative.")
```

### `jatmo_synthetic`

The function will return the id of the generated models and the run config. You can pass it a one-shot example. 

```
from jatmo import jatmo_synthetic

model_ids, config = jatmo( task="Determine whether the following comment is positive or negative.", one_shot_example = "This movie is awesome!")
```

## Common resources

You can use `from prompt_injection_defense.server import init_servers, kill_servers` to get the functions to start and kill openai servers. 
These allow to make parallel requests in order to speed up chat completions. 
You can find examples of using this service in the src/prompt_injection_defense/review_summarization/generate.py file. 

You can use `from prompt_injection_defense.server import rate_completions` to run the rating algorithm (using GPT-3.5-turbo). 
The docstring in the `prompt_injection_defense.server` file provides information as to its usage.

## Results

|                               | Code summarization | Sentiment Analysis | Review Summarization | Translation        | News Summarization | Toxicity Detection | Toxicity Detection (w/ GPT-generated label) | Sentence Similarity |
|-------------------------------|--------------------|--------------------|----------------------|--------------------|--------------------|--------------------|---------------------|---------------------|
| Dataset                       | The Stack          | IMDB Reviews       | Amazon Reviews       | Gutenberg Project  | CNN/DM             | [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)                   |                     |  [Semantic Textual Similarity Benchmark](https://github.com/PhilipMay/stsb-multi-mt)
| FT training size / test size  |                    |                    | 400 / 100            | 1000 / 100         | 400 / 100          | 400 / 100  | 400 / 100                   |  400/100                   |
|                    Base Model | curie              |   curie            |       davinci_002    |        davinci_002 |       davinci_002  | davinci_002 | davinci_002                   |   davinci_002                  |
| Quality vs Baseline           |                    | Better than GPT    | No Quality Loss      | No Quality Loss    | No Quality Loss      | Better than GPT (87%->92%)                   | No Quality Loss (86%)                    |  No Quality Loss 
| Success Rate PI [Start/GPT]   | 98%                | 100%               | 98%                  | 100%               | 99%                | 89% | 89%                   |                     |
| Success Rate PI [Start/FT]    | 0%                 | 0%                 | 0%                   | 0%                 | 0%                 | 0% | 0%                |                     |
| Success Rate PI [End/GPT]     | 96%                | 99%                | 100%                 | 100%               | 100%               | 84% | 84%                   | 99%                    |
| Success Rate PI [End/FT]      | 0%                 | 0%                 | 2%                   | 0%                 | 0%                 | 0% | 0%                |  0%                   |
| Success Rate PI [Middle/GPT]  | 12%                | 89%                | 93%                  | 52%                | 71%                | 85% | 85%                   |                     |
| Success Rate PI [Middle/FT]   | 0%                 | 0%                 | 0%                   | 0%                 | 0%                 | 0% | 0%                   |                     |
