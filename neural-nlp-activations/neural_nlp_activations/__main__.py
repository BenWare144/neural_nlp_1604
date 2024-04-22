import argparse
import fire
import logging
import sys
from datetime import datetime

from neural_nlp_activations import record_activations as record_activations_function

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")
for ignore_logger in ['transformers.data.processors', 'botocore', 'boto3', 'urllib3', 's3transfer']:
    logging.getLogger(ignore_logger).setLevel(logging.INFO)


def run(model, layers=None, subsample=None):
    start = datetime.now()
    record_activations_function(model=model, layers=layers, subsample=subsample)
    end = datetime.now()
    print(f"Duration: {end - start}")
    # print(score)


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    fire.Fire(command=FIRE_FLAGS)
