import os
import sys
import argparse
import glob
import logging
import copy
import yaml

import ROOT
from ROOT import TFile, TH1F, TH2F, TCanvas, gROOT
import CMS_lumi
import tdrstyle

import Classes
from Classes import Plot_TH1, Plot_TH2, Plot_Ratio_TH1, Plot_Multi_TH1, Plot_ROC, LoopPlotOnCanvas, MakeROCPlot, ProcessYAML, Plot_Multi_ROC, MakeMultiROCPlot

gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = 2000#[ROOT.kPrint, ROOT.kInfo]#, kWarning, kError, kBreak, kSysError, kFatal;

def main():
    #############################################################################################
    # Options #
    #############################################################################################
    parser = argparse.ArgumentParser(description='From given set of root files, make different histograms in a root file')
    parser.add_argument('-m','--model', action='store', required=True, type=str, default='',
                  help='NN model to be used')
    parser.add_argument('-v','--verbose', action='store_true', required=False, default=False,
            help='Show DEGUG logging')
    opt = parser.parse_args() 

    # Logging #
    if opt.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s') 
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    #############################################################################################
    # Select samples #
    #############################################################################################
    INPUT_DIR = os.path.join('/home/ucl/cp3/fbury/scratch/MoMEMta_output/NNOutput',opt.model)
    logging.info('Taking inputs from %s'%INPUT_DIR)
    
    # For each directory, puth the path in dictn the value is the list of histograms #
    class Plots():
        def __init__(self,name,override_params):
            self.name = name                            # Name of the subdir
            self.path = os.path.join(INPUT_DIR,name)    # Full path to files
            self.override_params = override_params      # Parameters to override in the configs
            self.list_histo = []                        # List of histograms to be filled

    list_plots = [
                    Plots(name = 'file_directory_name',
                          override_params = {}),
                 ]

    # Select template #
    class Template:
        def __init__(self,tpl,class_name):
            self.tpl = tpl                          # Template ".yml.tpl" to be used
            self.class_name = class_name            # Name of one of the class to use (TH1,TH2,...)

    templates = [
                    Template(tpl = 'TH1_template.yml.tpl',
                             class_name = 'Plot_TH1'),
                    Template(tpl = 'TH1Ratio_template.yml.tpl',
                             class_name = 'Plot_Ratio_TH1'),
                    Template(tpl = 'TH1Multi_template.yml.tpl',
                             class_name = 'Plot_Multi_TH1'),
                    Template(tpl = 'TH2_template.yml.tpl',
                             class_name = 'Plot_TH2'),
                ] 

    # Select template #
    class ROC:
        def __init__(self,tpl,class_name,def_name,plot_name,title):
            self.tpl = tpl                          # Template ".yml.tpl" to be used containing all the ROC curves parameters (they will all be plotted in the same figure)
            self.class_name = class_name            # Name of one of the class to use (ROC or multiROC)
            self.def_name = def_name                # Name of of the processing function 
            self.plot_name = plot_name              # Name of the plot -> .png
            self.title = title                      # Title of the plot
            self.list_instance = []
        def AddInstance(self,instance):
            self.list_instance.append(instance) # Contains the ROC configs listed in the tpl file

    rocs = [
                ROC(tpl = 'ROC_template.yml.tpl',
                    class_name = 'Plot_ROC',
                    def_name = 'MakeROCPlot',
                    plot_name = '',
                    title = '')
            ]

    # Make the output dir #
    OUTPUT_PDF = os.path.join(os.getcwd(),'PDF',opt.model)
    if not os.path.exists(OUTPUT_PDF):
        os.makedirs(OUTPUT_PDF)

    #############################################################################################
    # Loop over the different paths #
    #############################################################################################
    # Loop over the files #
    for obj in list_plots:
        logging.info('Starting Plotting from %s subdir'%obj.name)
        files = glob.glob(obj.path+'/*.root')

        if len(files) == 0:
            logging.error('Could not find %s at %s'%(obj.name,obj.path))

       # Instantiate all the ROCs #
        for roc in rocs:
            logging.debug('ROC template "%s" -> Class "%s" and process function %s '%(roc.tpl, roc.class_name, roc.def_name))
            YAML = ProcessYAML(roc.tpl) # Contain the ProcessYAML objects
            YAML.Particularize()
            for name,config in YAML.config.items():
                class_ = getattr(Classes, roc.class_name)
                logging.info('\tInitializing ROC %s'%(name))
                instance = class_(**config)
                roc.AddInstance(instance)

        # Loop over files #
        for f in files:
            #fullname = f.replace(obj.path+'/','').replace('.root','')
            fullname = os.path.basename(f).replace('.root','')
            logging.info('Processing weights from %s'%(fullname+'.root'))
            if fullname.startswith('DY'):
                filename = 'Drell-Yan'
            elif fullname.startswith('TT'):
                filename = 't#bar{t}'
            elif fullname.startswith('HToZA'):
                filename = 'Signal'
            else:
                filename = fullname
            ##############  ROC  section ################ 
            for roc in rocs:
                for inst_roc in roc.list_instance:
                    try:
                        valid = inst_roc.AddToROC(f)
                        if valid:
                            logging.info('\tAdded to ROC %s'%(inst_roc.title))
                    except Exception as e:
                        logging.warning('Could not add to ROC due to "%s"'%(e))
            
            ##############  HIST section ################ 
            # Loop over the templates #
            for template in templates: 
                logging.debug('Hist template "%s" -> Class "%s"'%(template.tpl, template.class_name))
                list_config = [] # Will contain the dictionaries of parameters
                YAML = ProcessYAML(template.tpl) # Contain the ProcessYAML objects
                params = {**{'filename':f,'title':filename},**obj.override_params}
                # Get the list of configs #
                YAML.Particularize(fullname)
                YAML.Override(params)
                # loop over the configs #
                for name,config in YAML.config.items():
                    try:
                        class_ = getattr(Classes, template.class_name)
                        logging.info('\tPlot %s'%(name))
                        instance = class_(**config)
                        instance.MakeHisto()
                        obj.list_histo.append(instance)
                    except Exception as e:
                        logging.warning('Could not plot %s due to "%s"'%(name,e))

        
        # Process ROCs #
        for roc in rocs:
            for inst_roc in roc.list_instance:
                try:
                    inst_roc.ProcessROC() 
                    logging.info('\tProcessed ROC %s'%(inst_roc.title))
                except Exception as e:
                    logging.warning('Could not process ROC due to "%s"'%(e))

        # Make ROCs graphs #
        for roc in rocs:
            try:
                def_ = getattr(Classes, roc.def_name)
                def_(roc.list_instance,name=os.path.join(OUTPUT_PDF,roc.plot_name),title=roc.title)
            except Exception as e:
                logging.warning('Could not plot ROC due to "%s"'%(e))

    #############################################################################################
    # Save histograms #
    #############################################################################################
    for obj in list_plots:
        PDF_name = os.path.join(OUTPUT_PDF,obj.name) 
        LoopPlotOnCanvas(PDF_name,obj.list_histo)
    logging.info("All Canvas have been printed")

if __name__ == "__main__":
    main()   
