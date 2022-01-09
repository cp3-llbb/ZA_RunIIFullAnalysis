import os, os.path
import json
import logging
logger = logging.getLogger("ZA-sys")
from bamboo import treefunctions as op

def  get_HLTZvtxSF(era=None, sample=None, uname=None, split_eras=False):
    #2016 :       1.0±0.0     
    #2017B :      0.934±0.005 
    #2017C :      0.992±0.001 
    #2017DEF :    1.000 
    #2017BCDEF :  0.991±0.001 
    #2018 :       1.0±0.0 
    if '2017' in era:
        if split_eras:
            if '2017B' in sample:
                HLTZvtxSF = op.systematic(op.c_float(0.934), name=uname, up=op.c_float(0,939), down=op.c_float(0,929))
            if  '2017C' in sample:
                HLTZvtxSF = op.systematic(op.c_float(0.992), name=uname, up=op.c_float(0.991), down=op.c_float(0.993))
            if '2017D' in sample or '2017E' in sample or '2017F' in sample:
                HLTZvtxSF = op.systematic(op.c_float(1.0), name=uname, up=op.c_float(1.0), down=op.c_float(1.0))
        else:
            HLTZvtxSF = op.systematic(op.c_float(0.991), name=uname, up=op.c_float(0.992), down=op.c_float(0.990))
    else:
        HLTZvtxSF = op.systematic(op.c_float(1.0), name=uname, up=op.c_float(1.0), down=op.c_float(1.0))
    return HLTZvtxSF

def get_tthDYreweighting(era, jets):
#def getDYReweighting(era, sample, jets, bjets, WP):
    # to be applied only on the NLO DY+jets samples
    #if WP =='M': # Let's focus on the Medium Working Point 
    if op.rng_len(jets) >= 4:
    #if op.AND(op.rng_len(jets) >= 4, op.rng_len(bjets) >= 2):
        if era == '2017':
            nominal = 1.453
            uncer = 0.081
        elif era == '2018':
            nominal = 1.329
            uncer = 0.140

    elif op.rng_len(jets) == 3:
    #elif op.AND(op.rng_len(jets) == 3 , op.rng_len(bjets) >= 2):
        if era == '2017':
            nominal = 1.054
            uncer = 0.036
        elif era == '2018':
            nominal = 1.012
            uncer = 0.046

    elif op.rng_len(jets) == 2:
    #elif op.AND(op.rng_len(jets) == 2 , op.rng_len(bjets) >= 2):
        if era == '2017':
            nominal = 0.0884
            uncer = 0.033
        elif era == '2018':
            nominal = 0.822
            uncer = 0.021

    return op.systematic(op.c_float(nominal), name="NLO_DYReweighting", up=op.c_float(nominal+ uncer), down=op.c_float(nominal- uncer))

def get_HLTsys(era, leptons, suffix, version):
    # taken from ttH studies:
    # https://gitlab.cern.ch/ttH_leptons/doc/blob/master/Legacy/data_to_mc_corrections.md#trigger-efficiency-scale-factors
    if version == "tth":
        if era == "2016":
            if suffix == "MuMu":
                # SF = 1.010   +/-    0.010
                wgt = op.systematic(op.c_float(1.010), name="mumutrig", up=op.c_float(1.020), down=op.c_float(1.0))
            elif suffix == "ElEl":
                # SF = 1.020   +/-    0.020
                wgt = op.systematic(op.c_float(1.020), name="eleltrig", up=op.c_float(1.040), down=op.c_float(1.0))
            else:
                # SF = 1.020   +/-    0.010
                wgt = op.systematic(op.c_float(1.020), name="{0}trig".format(suffix.lower()), up=op.c_float(1.030), down=op.c_float(1.010))
        
        elif era == "2017":
            ll_pt = (leptons[0].p4 + leptons[1].p4).Pt()
            if suffix =="MuMu":
                if ( ll_pt < 35.):
                    # SF = 0.972   +/-    0.006
                    wgt = op.systematic(op.c_float(0.972), name="mumutrig", up=op.c_float(0.978), down=op.c_float(0.966))
                elif (ll_pt >= 35.):
                    # SF = 0.994   +/-    0.001
                    wgt = op.systematic(op.c_float(0.994), name="mumutrig", up=op.c_float(0.995), down=op.c_float(0.993))
            
            elif suffix == "ElEl":
                if (ll_pt < 30.):
                    # SF = 0.937   +/-    0.027
                    wgt = op.systematic(op.c_float(0.937), name="eleltrig", up=op.c_float(0.964), down=op.c_float(0.91))
                elif (ll_pt >= 30.):
                    # SF = 0.991   +/-    0.002
                    wgt = op.systematic(op.c_float(0.991), name="eleltrig", up=op.c_float(0.993), down=op.c_float(0.989))

            else:
                if (ll_pt < 35.):
                    # SF = 0.952   +/-    0.008
                    wgt = op.systematic(op.c_float(0.952), name="{0}trig".format(suffix.lower()), up=op.c_float(0.96), down=op.c_float(0.944))
                elif (op.in_range(35., ll_pt, 50)):
                    # SF = 0.983   +/-    0.003
                    wgt = op.systematic(op.c_float(0.983), name="{0}trig".format(suffix.lower()), up=op.c_float(0.986), down=op.c_float(0.980))
                elif (ll_pt >= 50.):
                    # SF = 1.000   +/-    0.001
                    wgt = op.systematic(op.c_float(1.000), name="{0}trig".format(suffix.lower()), up=op.c_float(0.001), down=op.c_float(0.999))
        else:
            raise RuntimeError("Trigger SFs for era {0} still missing in tth analysis---> pass to other version ! ".format(era)) 

    elif version == "CMS_AN-19-140":
        # https://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_140_v2.pdf
        if era == "2016":
            if suffix == "MuMu":
                # SF = 0.996 ± 0.006
                wgt = op.systematic(op.c_float(0.996), name="mumutrig", up=op.c_float(1.002), down=op.c_float(0.99))
            elif suffix == "ElEl":
                # SF = 0.992 ± 0.006
                wgt = op.systematic(op.c_float(0.992), name="eleltrig", up=op.c_float(0.998), down=op.c_float(0.986))
            else:
                # SF = 0.996 ± 0.006
                wgt = op.systematic(op.c_float(0.996), name="{0}trig".format(suffix.lower()), up=op.c_float(1.002), down=op.c_float(0.99))
        elif era == "2017":
            if suffix == "MuMu":
                # SF = 0.987 ± 0.005
                wgt = op.systematic(op.c_float(0.987), name="mumutrig", up=op.c_float(0.992), down=op.c_float(0.982))
            elif suffix == "ElEl":
                # SF = 0.975 ± 0.005
                wgt = op.systematic(op.c_float(0.975), name="eleltrig", up=op.c_float(0.98), down=op.c_float(0.97))
            else:
                # SF = 0.985 ± 0.005
                wgt = op.systematic(op.c_float(0.985), name="{0}trig".format(suffix.lower()), up=op.c_float(0.99), down=op.c_float(0.98))
        else:
            if suffix == "MuMu":
                # SF = 0.990 ± 0.005
                wgt = op.systematic(op.c_float(0.990), name="mumutrig", up=op.c_float(0.995), down=op.c_float(0.985))
            elif suffix == "ElEl":
                # SF = 0.989 ± 0.005
                wgt = op.systematic(op.c_float(0.989), name="eleltrig", up=op.c_float(0.994), down=op.c_float(0.984))
            else:
                # SF = 0.992 ± 0.005
                wgt = op.systematic(op.c_float(0.992), name="{0}trig".format(suffix.lower()), up=op.c_float(0.997), down=op.c_float(0.987))
    return wgt

def get_POG_highPT_MU_RECO_EFF(mu_mom, mu_eta, corr_file, era):
    with open(corr_file) as f:
        corrections = json.load(f)

    nominal    = 1. 
    error_high = 0.
    error_low  = 0.

    for i in range(len(corrections['data'])):
        print( "*/*", "eta:", corrections['data'][i]['bin'])
        eta_min = corrections['data'][i]['bin'][0]
        eta_max = corrections['data'][i]['bin'][1]
        while  op.in_range(eta_min, mu_eta, eta_max):
            momentum_min = corrections['data'][i]['bin'][0] 
            momentum_max = corrections['data'][i]['bin'][1]
            while j in range(len(corrections['data'][i]['values'])):
                print ( "momentum :", corrections['data'][i]['values'][j]['bin'], "values : ", corrections['data'][i]['values'][j])
                
                if op.in_range(momentum_min, mu_mom, momentum_max):

                    nominal     = corrections['data'][i]['values'][j]['value']
                    error_high  = corrections['data'][i]['values'][j]['error_high']
                    error_low   = corrections['data'][i]['values'][j]['error_low']
                    break
    return op.systematic(op.c_float(nominal), name="TkMuonHighpTReco", up=op.c_float(error_high), down=op.c_float(error_low))
