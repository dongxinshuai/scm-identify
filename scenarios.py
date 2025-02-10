from DGM.LinearSCM import LinearSCM
from DGM.DataModel import DataModel
import pandas as pd
import functools
import numpy as np
import os 

def teacher_burnout(normalized=False, seed=0):

    df=pd.read_csv('Data/Teacher_Burnout_data.csv')
    rename_dict = {k:'X_'+k for k in df.columns}
    df = df.rename(columns=rename_dict)

    if normalized:
        df=(df-df.mean())
        df=df/df.std()

    dgm = DataModel(df, df)

    return dgm 

def multitasking(normalized=False, seed=0):


    df = pd.read_csv('Data/Final_Multitasking_Data.csv')
    columns = ['Speed_Number', 'Speed_Letter', 'Speed_Figure', 'Error_Number',
    'Error_Letter', 'Error_Figure', 'AverageQus_Par1', 'AverageQus_Par2',
    'AverageQus_Par3']

    df = df[columns]
    rename_dict = {k:'X_'+k for k in df.columns}
    df = df.rename(columns=rename_dict)

    if normalized:
        df=(df-df.mean())
        df=df/df.std()

    dgm = DataModel(df, df)

    return dgm 

def LLCM_LLH_Case1(seed=0):
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("L4", False)
    dgm.add_variable("X1", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("L5", False)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)
    dgm.add_variable("X13", True)
    dgm.add_variable("X14", True)
    dgm.add_variable("X15", True)
    dgm.add_variable("X16", True)
    dgm.add_variable("X17", True)

    dgm.add_edge("L1", "L3")
    dgm.add_edge("L2", "L3")
    dgm.add_edge("L1", "L4")
    dgm.add_edge("L2", "L4")
    dgm.add_edge("L1", "X1")
    dgm.add_edge("L2", "X1")
    dgm.add_edge("L1", "X2")
    dgm.add_edge("L2", "X2")
    dgm.add_edge("L1", "X3")
    dgm.add_edge("L2", "X3")
    dgm.add_edge("L1", "L5")
    dgm.add_edge("L2", "L5")
    dgm.add_edge("L1", "X4")
    dgm.add_edge("L2", "X4")
    dgm.add_edge("L1", "X5")
    dgm.add_edge("L2", "X5")

    dgm.add_edge("L3", "X6")
    dgm.add_edge("L4", "X6")
    dgm.add_edge("X1", "X6")
    dgm.add_edge("L3", "X7")
    dgm.add_edge("L4", "X7")
    dgm.add_edge("X1", "X7")
    dgm.add_edge("L3", "X8")
    dgm.add_edge("L4", "X8")
    dgm.add_edge("X1", "X8")
    dgm.add_edge("L3", "X9")
    dgm.add_edge("L4", "X9")
    dgm.add_edge("X1", "X9")
    dgm.add_edge("L3", "X10")
    dgm.add_edge("L4", "X10")
    dgm.add_edge("X1", "X10")
    dgm.add_edge("L3", "X11")
    dgm.add_edge("L4", "X11")
    dgm.add_edge("X1", "X11")

    dgm.add_edge("X2", "X12")
    dgm.add_edge("X2", "X13")

    dgm.add_edge("X3", "X14")

    dgm.add_edge("L5", "X15")
    dgm.add_edge("L5", "X16")

    dgm.add_edge("X4", "X17")
    dgm.add_edge("X5", "X17")

    return dgm

def LLCM_LLH_Case2(seed=0):
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("L4", False)

    dgm.add_variable("X1", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)
    dgm.add_variable("X13", True)
    dgm.add_variable("X14", True)
    dgm.add_variable("X15", True)
    dgm.add_variable("X16", True)

    dgm.add_edge("L1", "L2")
    dgm.add_edge("L1", "X2")
    dgm.add_edge("L1", "L3")
    dgm.add_edge("L1", "X3")
    dgm.add_edge("L1", "L4")
    dgm.add_edge("L1", "X4")
    dgm.add_edge("L1", "X5")

    dgm.add_edge("X1", "L2")
    dgm.add_edge("X1", "X2")
    dgm.add_edge("X1", "L3")
    dgm.add_edge("X1", "X3")
    dgm.add_edge("X1", "L4")
    dgm.add_edge("X1", "X4")
    dgm.add_edge("X1", "X5")

    dgm.add_edge("L2", "X6")
    dgm.add_edge("L2", "X7")
    dgm.add_edge("L2", "X8")
    dgm.add_edge("L2", "X9")
    dgm.add_edge("L2", "X10")

    dgm.add_edge("X2", "X6")
    dgm.add_edge("X2", "X7")
    dgm.add_edge("X2", "X8")
    dgm.add_edge("X2", "X9")
    dgm.add_edge("X2", "X10")

    dgm.add_edge("L3", "X10")
    dgm.add_edge("L3", "X11")
    dgm.add_edge("L3", "X12")

    dgm.add_edge("X3", "X13")

    dgm.add_edge("L4", "X14")
    dgm.add_edge("L4", "X15")

    dgm.add_edge("X4", "X16")
    dgm.add_edge("X5", "X16")

    return dgm


def LLCM_MM_Case1(seed=0):
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("L4", False)

    dgm.add_variable("X1", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)

    dgm.add_edge("L1", "X1")
    dgm.add_edge("L1", "X2")
    dgm.add_edge("L1", "L2")
    dgm.add_edge("L1", "L3")

    dgm.add_edge("L2", "X3")
    dgm.add_edge("L2", "X4")
    dgm.add_edge("L2", "L4")

    dgm.add_edge("L3", "X5")
    dgm.add_edge("L3", "X6")
    dgm.add_edge("L3", "X7")
    dgm.add_edge("L3", "L4")

    dgm.add_edge("L4", "X8")
    dgm.add_edge("L4", "X9")

    dgm.add_edge("X5", "X10")
    dgm.add_edge("X5", "X11")
    dgm.add_edge("X5", "X12")

    return dgm


def LLCM_MM_Case2(seed=0):
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)

    dgm.add_variable("X1", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)

    dgm.add_edge("X1", "X2")
    dgm.add_edge("X1", "L1")
    dgm.add_edge("X1", "L2")

    dgm.add_edge("L1", "X3")
    dgm.add_edge("L1", "X4")
    dgm.add_edge("L2", "X5")
    dgm.add_edge("L2", "X6")

    dgm.add_edge("L1", "X7")
    dgm.add_edge("L2", "X7")
    
    dgm.add_edge("X7", "X8")
    

    return dgm

def LLCM_Tree_Case1(seed=0):
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("X1", True)
    dgm.add_variable("L1", False)
    dgm.add_variable("X2", True)
    dgm.add_variable("L2", False)

    dgm.add_edge("X1", "L1")
    dgm.add_edge("X1", "X2")
    dgm.add_edge("X1", "L2")

    dgm.add_variable("X3", True)
    dgm.add_variable("L3", False)
    dgm.add_variable("X4", True)

    dgm.add_edge("L1", "X3")
    dgm.add_edge("L1", "L3")
    dgm.add_edge("L1", "X4")

    dgm.add_variable("X5", True)
    dgm.add_variable("L4", False)
    dgm.add_variable("X6", True)

    dgm.add_edge("X2", "X5")
    dgm.add_edge("X2", "L4")
    dgm.add_edge("X2", "X6")

    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)

    dgm.add_edge("L2", "X7")
    dgm.add_edge("L2", "X8")
    dgm.add_edge("L2", "X9")

    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)

    dgm.add_edge("L3", "X10")
    dgm.add_edge("L3", "X11")
    dgm.add_edge("L3", "X12")

    dgm.add_variable("X13", True)
    dgm.add_variable("X14", True)
    dgm.add_variable("X15", True)

    dgm.add_edge("L4", "X13")
    dgm.add_edge("L4", "X14")
    dgm.add_edge("L4", "X15")

    dgm.add_variable("X16", True)
    dgm.add_variable("X17", True)
    dgm.add_variable("X18", True)

    dgm.add_edge("X8", "X16")
    dgm.add_edge("X8", "X17")
    dgm.add_edge("X8", "X18")

    return dgm


dgm_by_scenario = {
    "LLCM_LLH_Case1": LLCM_LLH_Case1(), 
    "LLCM_LLH_Case2": LLCM_LLH_Case2(),
    "LLCM_MM_Case1": LLCM_MM_Case1(),
    "LLCM_MM_Case2": LLCM_MM_Case2(),
    "LLCM_Tree_Case1": LLCM_Tree_Case1(),
    "multitasking": multitasking(),
    "teacher_burnout": teacher_burnout(),
}

