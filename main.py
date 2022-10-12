97-122

from collections import namedtuple
import collections
import random
from re import I
from typing import Counter
import readchar
from colorama import Fore, Style
import sys
import argparse
import time
from pprint import pprint


def gen_random():
    #gera um número aleatório entre os codigos ascii de 'a' e 'z'
    val= random.randint(97,122)
    return val

def gen_key():
    #Gera um char aleatório [a-z] ,espera a resposta do utilizador e dá feedback visual sobre a mesma

    randnum= gen_random()

    print("Type letter "+chr(randnum))
    start=time.time()
    while True:
        try:
            key=ord(readchar.readkey())
            break
        except:
            #necessário caso seja premido "PG Up" por exemplo
            print("Invalid key press, try again. Please use only characters")

    if key==randnum:
        print("You typed key "+Fore.GREEN+chr(key)+Style.RESET_ALL)
        
    elif key==32:
        sys.exit()
    
    else:
        print("You typed key "+Fore.RED+chr(key)+Style.RESET_ALL)

    I= KeyPress(chr(randnum),chr(key),time.time()-start)
    return I

def define_args():
    #Define e trata de todos os argumentos vindos da consola

    parser = argparse.ArgumentParser(prog='main' ,description='Definition of test mode')
    parser.add_argument("-utm", "--use_time_mode", action='store_true', default=False,help="Max number of secs for time mode or maximum number of inputs for number of inputs mode.")
    parser.add_argument("-mv", "--max_value",type=int, help="Max number of seconds for time mode or maximum number of inputs for number of inputs mode.")
    args= parser.parse_args()
    return args


def generate_report(inputlist,timer,test_start,test_end):
    #Gera um dicionario a partir de uma lista de inputs, duração do teste e a data de início e fim do teste 
    hit_counter=0
    miss_counter=0
    type_total_duration=0
    type_hit_total_duration=0
    type_miss_total_duration=0
    #Calculo dos valores do dicionario final
    for i in inputlist:
        if i.requested== i.received:
            hit_counter+=1
            type_hit_total_duration+=i.duration
        else:
            miss_counter +=1
            type_miss_total_duration+=i.duration
        type_total_duration+=i.duration
    accuracy=hit_counter/len(inputlist)
    test_duration=timer
    test_end=test_end
    test_start=test_start
    type_miss_average_duration=type_miss_total_duration/miss_counter
    type_hit_average_duration=type_hit_total_duration/hit_counter
    type_average_duration= type_total_duration/len(inputlist)

    #Geração do dicionario final
    report={}
    report["accuracy"]=accuracy
    report["number_of_hits"]=hit_counter
    report["number_of_types"]=len(inputlist)
    report["test_duration"]=test_duration
    report["test_end"]=test_end
    report["test_start"]=test_start
    report["type_average_duration"]=type_average_duration
    report["type_hit_average_duration"]=type_hit_average_duration
    report["type_miss_average_duration"]=type_miss_average_duration
    report["types"]=inputlist



    return report



def main():
    global KeyPress
    args=define_args()
    print(vars(args))
    #Verificar a existência de argumentos
    try:
        check= sys.argv[1]
    except:
        print("Program requires arguments, run with -h for more information")
        sys.exit()
    
    #Valores vindos dos argumentos
    timed_run_flag= vars(args)["use_time_mode"]
    max_value=vars(args)["max_value"]
    

    counter=0
    KeyPress= collections.namedtuple('KeyPress',["requested","received","duration"])
    inputlist=[]

    if timed_run_flag:
        print("Test running up to "+ str(max_value)+" seconds.")
    else:
        print("Test running up to "+ str(max_value)+" inputs.")
        
    print("Press any key to start the test")
    
    readchar.readkey() #inicialização do teste
    
    startgentimer= time.time()
    test_start=time.strftime("%a, %b %d %Y %H:%M:%S %Y")

    #se for temporizado
    if timed_run_flag:
        while time.time()-startgentimer<max_value:
            counter+=1
            inputlist.append(gen_key())
        
        print("Current test duration {} exceeds maximum of {}".format((time.time()-startgentimer),max_value))
    #se não for 
    else:
        
        while counter<max_value:
            counter+=1
            inputlist.append( gen_key())
    #conclusão do teste
    test_end=time.strftime("%a, %b %d %Y %H:%M:%S %Y")
    print(Fore.BLUE+"Test finished!"+Style.RESET_ALL)

    report=generate_report(inputlist,time.time()-startgentimer,test_start,test_end)
    pprint(report)
    
if __name__ == '__main__':

    main()