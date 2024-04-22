import argparse
import fire
import logging
import sys
from datetime import datetime

from neural_nlp import score as score_function

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")
for ignore_logger in ['transformers.data.processors', 'botocore', 'boto3', 'urllib3', 's3transfer']:
    logging.getLogger(ignore_logger).setLevel(logging.INFO)


def run(benchmark, model, layers=None, subsample=None):
    start = datetime.now()

    time_stamp=start
    scores_fn=f"/home/ben/data/scores/{model}_{benchmark}_{time_stamp}_score_raw"
    print("scores_fn:",scores_fn)

    score = score_function(model=model, layers=layers, subsample=subsample, benchmark=benchmark)
    end = datetime.now()
    print(f"Duration: {end - start}")
    print(score)
    with open(f"{scores_fn}.txt", 'w') as f:
        f.write(str(score))
    df=score.to_dataframe(name=f"{scores_fn}")
    df.to_csv(f"{scores_fn}.csv")



if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    fire.Fire(command=FIRE_FLAGS)
