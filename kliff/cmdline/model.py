import sys

from kliff.models import KIM


def get_avail_params(model_name):
    model = KIM(model_name)
    model.echo_model_params()


class Command:
    """Utility to manipulate dataset.
    """

    @staticmethod
    def add_arguments(parser):
        func = parser.add_argument
        func(
            "-a",
            "--avail-params",
            type=str,
            metavar="model_name",
            help="Get the available parameters of a KIM model to be used for fitting.",
        )

    @staticmethod
    def run(args, parser):

        if args.avail_params is not None:
            get_avail_params(args.avail_params)
        else:
            parser.print_help()


if __name__ == "__main__":

    model_name = "SW_StillingerWeber_1985_Si__MO_405512056662_005"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    get_avail_params(model_name)
