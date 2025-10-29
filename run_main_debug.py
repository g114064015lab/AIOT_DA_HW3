"""Debug runner that captures full logs to file."""
import logging
from pathlib import Path

import main

if __name__ == '__main__':
    # Set up file logging
    log_path = Path('run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting debug run...")
    try:
        main.main()
        logging.info("Run completed successfully.")
    except Exception as e:
        logging.exception("Run failed with error: %s", e)
        raise