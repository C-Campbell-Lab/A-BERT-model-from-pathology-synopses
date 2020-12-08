import logging
import os

from .web import Server, WebConfig


def main(reset=False):
    logger = logging.getLogger()
    os.makedirs("data", exist_ok=True)
    web_config = WebConfig(
        "dataset.zip", "model", "data/unlabelled.json", "data/unstate.pkl"
    )
    if reset:
        logger.warning("RESET")
        web_config.reset()
    web_config.init_static()

    server = Server(web_config)
    server.plot()
    server.app.run_server(debug=False)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
