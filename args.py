import  argparse ,os, sys 
import  enum 

class  classeEnum(enum.Enum):
    verde="green"
    blu="blue"
    rosso="red"


print  (" inizio")

parser=argparse.ArgumentParser(description="descrizione molto importante")
parser.add_argument("-t","--test",action='store_true',default=False)
parser.add_argument("-e", "--enum", choices=[e.value for e in classeEnum],default="red", help="scegli un colore")
parser.add_argument("-val","--valore",dest="valore_str",type=str,help="  devi mettere  il valore stringa ",required=True)
args=parser.parse_args()


print(args.valore_str)
print(args.enum)
print(classeEnum.blu)