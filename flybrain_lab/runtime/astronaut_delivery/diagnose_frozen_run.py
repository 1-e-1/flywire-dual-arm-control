"""Run the unchanged controller with periodic Python stack evidence."""
import argparse,faulthandler,runpy,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace-after',type=float,default=45);p.add_argument('controller_args',nargs=argparse.REMAINDER)
    args=p.parse_args();rest=args.controller_args
    if rest[:1]==['--']:rest=rest[1:]
    sys.argv=['astronaut_delivery.sufficiency_delivery',*rest]
    faulthandler.dump_traceback_later(args.trace_after,repeat=True)
    try:runpy.run_module('astronaut_delivery.sufficiency_delivery',run_name='__main__')
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
