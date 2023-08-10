"""Example with DeepDrug."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import DeepDrug
import argparse
import torch
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--precision', type=str, default='float32', help='precision')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument('--batch_size', type=int, default=-1, help='batch_size')
parser.add_argument('--num_iter', type=int, default=-1, help='num_iter')
parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')
parser.add_argument('--profile', dest='profile', action='store_true', help='profile')
parser.add_argument('--quantized_engine', type=str, default=None, help='quantized_engine')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate')
parser.add_argument('--ipex', dest='ipex', action='store_true', help='ipex')
parser.add_argument('--jit', dest='jit', action='store_true', help='jit')
args = parser.parse_args()
print(args)


def main():
    """Train and evaluate the EPGCNDS model."""
    dataset = DrugCombDB()
    model = DeepDrug()
    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.001),
        batch_size=args.batch_size,
        epochs=1,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    # results.summarize()

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'deepdrug-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == "__main__":

    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(args.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            args.p = p
            if args.precision == "bfloat16":
                print('---- Enable AMP bfloat16')
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    main()
            elif args.precision == "float16":
                print('---- Enable AMP float16')
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
                    main()
            else:
                main()
    else:
        if args.precision == "bfloat16":
            print('---- Enable AMP bfloat16')
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                main()
        elif args.precision == "float16":
            print('---- Enable AMP float16')
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
                main()
        else:
            main()
