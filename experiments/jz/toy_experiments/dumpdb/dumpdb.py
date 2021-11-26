import logging
from wikipedia2vec.dump_db import DumpDB
import argparse

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--path-to-dump", help="echo the string you use here")
args = parser.parse_args()

logger.info(f"using as args: {args}")

dump_db = DumpDB(args.path_to_dump)

logger.info("the loading of the database has succeed")

redirects_map = {key.lower(): value for key, value in dump_db.redirects()}

logger.info(f"`redirects_map` keys are {list(redirects_map.keys())}")

data = dump_db.get_paragraphs(redirects_map["america"]).text[0]
