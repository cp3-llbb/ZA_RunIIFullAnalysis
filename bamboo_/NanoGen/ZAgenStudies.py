import os
import sys
import logging
logger = logging.getLogger("ZA NanoGen")

from bamboo.analysismodules import NanoAODHistoModule
from bamboo.analysisutils import makePileupWeight

from bamboo.plots import Plot
from bamboo.plots import EquidistantBinning as EqBin
from bamboo import treefunctions as op

GENPath = os.path.dirname(__file__)
if GENPath not in sys.path:
    sys.path.append(GENPath)

import utils
def addTheorySystematics(tree, noSel, qcdScale=True, PSISR=True, PSFSR=True, PDFs=True):
    if qcdScale:
        qcdScaleVariations = { f"qcdScalevar{i}": tree.LHEScaleWeight[i] for i in [0, 1, 3, 5, 7, 8] }
        qcdScaleSyst = op.systematic(op.c_float(1.), name="qcdScale", **qcdScaleVariations)
        noSel = noSel.refine("qcdScale", weight=qcdScaleSyst)

    if PSISR :#and "PS" not in buggySyst:
        psISRSyst = op.systematic(op.c_float(1.), name="psISR", up=tree.PSWeight[2], down=tree.PSWeight[0])
        noSel = noSel.refine("psISR", weight=psISRSyst)
    
    if PSFSR :#and "PS" not in buggySyst:
        psFSRSyst = op.systematic(op.c_float(1.), name="psFSR", up=tree.PSWeight[3], down=tree.PSWeight[1])
        noSel = noSel.refine("psFSR", weight=psFSRSyst)
    
    if PDFs:
        pdfsWeight = op.systematic(tree.LHEPdfWeight[0], name="pdfWgt", up=op.rng_max(tree.LHEPdfWeight[1:]), down=op.rng_min(tree.LHEPdfWeight[1:]))

    return noSel

from bamboo.treedecorators import NanoAODDescription
nanoGenDescription = NanoAODDescription(
    groups=["HTXS_", "Generator_", "MET_", "GenMET_", "LHE_", "GenVtx_"],
    collections=["nGenPart", "nGenJet", "nGenJetAK8", "nGenVisTau", "nGenDressedLepton", "nGenIsolatedPhoton", "nLHEPart"]
)
class NanoGenHtoZAPlotter(NanoAODHistoModule):
    """ H->Z(ll)A(bb) Nano-Gen studies """
    def __init__(self, args):
        super(NanoGenHtoZAPlotter, self).__init__(args)
        self.plotDefaults = {
                            "y-axis"           : "Events",
                            "log-y"            : "both",
                            "y-axis-show-zero" : True,
                            "save-extensions"  : ["png"],
                            "show-ratio"       : True,
                            "sort-by-yields"   : False,
                            #"legend-columns"   : 2
                            "normalized"       : True # normalize to 1 
                            }
        self.doSysts = self.args.systematic
    def addArgs(self, parser):
        super(NanoGenHtoZAPlotter, self).addArgs(parser)
        parser.add_argument("-s", "--systematic", action="store_true", help="Produce systematic variations")
        parser.add_argument("--backend", type=str, default="dataframe", help="Backend to use, 'dataframe' (default) or 'lazy'")

    def prepareTree(self, tree, sample=None, sampleCfg=None):
        tree,noSel,be,lumiArgs = super(NanoGenHtoZAPlotter, self).prepareTree(tree, sample=sample, sampleCfg=sampleCfg, description=nanoGenDescription)
        if self.isMC(sample):
            # if it's a systematics sample, turn off other systematics
            if "syst" in sampleCfg:
                self.doSysts = False
            noSel = noSel.refine("mcWeight", weight=tree.genWeight, autoSyst=self.doSysts)
            if self.doSysts:
                noSel = utils.addTheorySystematics(self, sample, sampleCfg, tree, noSel)
        return tree,noSel,be,lumiArgs

    def definePlots(self, t, noSel, sample=None, sampleCfg=None):
        plots = []
        binScaling=1

        sorted_GenDressedLepton = op.sort(t.GenDressedLepton, lambda lep : -lep.pt)
        
        h3=op.select(t.GenPart,lambda obj: obj.pdgId == 36 )
        h2=op.select(t.GenPart,lambda obj: obj.pdgId == 35 )
        z = op.select(t.GenPart,lambda obj: obj.pdgId == 23)
        
        genMuons = op.select(sorted_GenDressedLepton, lambda l : op.AND( op.abs(l.pdgId) == 13, op.abs(l.eta) < 2.4, l.pt > 10.))
        genElectrons = op.select(sorted_GenDressedLepton, lambda l : op.AND(op.abs(l.pdgId) == 11, op.abs(l.eta) < 2.5, l.pt > 15.))

        sorted_GenJet= op.sort(t.GenJet, lambda j : -j.pt)
        genJets = op.select(sorted_GenJet, lambda j : op.AND( j.pt > 20., op.abs(j.eta) < 2.4))
        genCleanedJets = op.select(genJets, lambda j: op.AND(
                            op.NOT(op.rng_any(genElectrons, lambda el: op.deltaR(j.p4, el.p4) < 0.4 )),
                            op.NOT(op.rng_any(genMuons, lambda mu: op.deltaR(j.p4, mu.p4) < 0.4 ))
                        ))

        sorted_GenJetAK8= op.sort(t.GenJetAK8, lambda j : -j.pt)
        genJetsAK8 = op.select(sorted_GenJetAK8, lambda j : op.AND(j.pt > 200., op.abs(j.eta) >2.4))

        genBJets = op.select(genCleanedJets, lambda jet: jet.hadronFlavour == 5)
        genLightJets = op.select(genCleanedJets, lambda jet: jet.hadronFlavour != 5)

        # H->ZA->ll bb
        Jets_dict = {"nGenJets": genJets,
                      "nBJets" : genBJets,
                      "nLightJets": genBJets}

        for key, Obs in Jets_dict.items():
            plots.append(Plot.make1D(f"{key}", op.rng_len(Obs), noSel,
                        EqBin(7, 0., 7.), title=f"{key}"))

        OSSFLeptons_Zmass = lambda lep1,lep2 : op.AND(lep1.pdgId == -lep2.pdgId, op.in_range(70., op.invariant_mass(lep1.p4, lep2.p4), 120.))
        OSSFLeptons   = lambda lep1,lep2 : op.AND(lep1.pdgId == -lep2.pdgId)
        LowMass_cut = lambda lep1, lep2: op.invariant_mass(lep1.p4, lep2.p4)>12.

        hasOSLL_cmbRng = lambda cmbRng : op.AND(op.rng_len(cmbRng) > 0, cmbRng[0][0].pt > 25.) 
        OSLeptons = { "mumu": op.combine(genMuons, N=2, pred= OSSFLeptons_Zmass),
                      "elel": op.combine(genElectrons, N=2, pred=OSSFLeptons_Zmass),
                      #"elmu": op.combine((genMuons, genElectrons), pred=lambda mu,ele : op.AND(LowMass_cut(mu, ele), mu.pdgId != -ele.pdgId)),
                    }

        TwoOSLeptonsSel = noSel.refine("hasOSLL", cut=op.OR(*( hasOSLL_cmbRng(rng) for rng in OSLeptons.values())))
        categories = dict((channel, (catLLRng[0], TwoOSLeptonsSel.refine("hasOs{0}".format(channel), cut=hasOSLL_cmbRng(catLLRng)))) for channel, catLLRng in OSLeptons.items())

        for channel, (TwoLeptons, catSel) in categories.items():
            TwoLeptonsTwoGenJets = catSel.refine("TwoGenJet_{0}Sel".format(channel), cut=[ op.rng_len(genCleanedJets) > 1])
            TwoLeptonsTwoGenBJets = catSel.refine("TwoGenBJet_{0}Sel".format(channel), cut=[ op.rng_len(genBJets) > 1])

            jj_p4 = genCleanedJets[0].p4
            lljj_p4 = (TwoLeptons[0].p4 +TwoLeptons[1].p4 + jj_p4)
            plots += [ Plot.make1D(f"{channel}_genjj{nm}", var, TwoLeptonsTwoGenJets, binning,
                    title=f"GEN {title}", plotopts=utils.getOpts(channel))
                    for nm, (var, binning, title) in {
                    "PT" : (jj_p4.Pt() , EqBin(60 // binScaling, 0., 450.), "jj P_{T} [GeV]"),
                    "Phi": (jj_p4.Phi(), EqBin(50 // binScaling, -3.1416, 3.1416), "jj #phi"),
                    "Eta": (jj_p4.Eta(), EqBin(50 // binScaling, -3., 3.), "jj Eta"),
                    "mjj": (jj_p4.M(), EqBin(60 // binScaling, 0., 850.), "M_{jj} [GeV]"),
                    "mlljj": (lljj_p4.M(), EqBin(60 // binScaling, 120., 1000.), "M_{lljj} [GeV]")
            }.items() ]

            bb_p4 = genBJets[0].p4
            llbb_p4 = (TwoLeptons[0].p4 +TwoLeptons[1].p4 + bb_p4)
            
            plots += [ Plot.make1D(f"{channel}_genbb{nm}", var, TwoLeptonsTwoGenBJets, binning,
                    title=f"GEN {title}", plotopts=utils.getOpts(channel))
                    for nm, (var, binning, title) in {
                    "PT" : (bb_p4.Pt() , EqBin(60 // binScaling, 0., 450.), "bb P_{T} [GeV]"),
                    "Phi": (bb_p4.Phi(), EqBin(50 // binScaling, -3.1416, 3.1416), "bb #phi"),
                    "Eta": (bb_p4.Eta(), EqBin(50 // binScaling, -3., 3.), "bb Eta"),
                    "mbb": (bb_p4.M(), EqBin(60 // binScaling, 0., 850.), "M_{bb} [GeV]"),
                    "mllbb": (lljj_p4.M(), EqBin(60 // binScaling, 120., 1000.), "M_{llbb} [GeV]")
            }.items() ]

            maxJet = 1
            for idx, sel in enumerate([TwoLeptonsTwoGenJets,TwoLeptonsTwoGenBJets]):
                jtype =('GenJet' if idx ==0 else( 'GenBJet'))
                eqBin_pt = EqBin(60 // binScaling, 0., 650.)
                for i in range(maxJet):
                    plots += [ Plot.make1D(f"{channel}_{jtype}{i+1}_{nm}",
                            jVar(genCleanedJets[i]), sel, binning, title=f"{utils.getCounter(i+1)} {jtype} {title}",
                            plotopts=utils.getOpts(channel))
                        for nm, (jVar, binning, title) in {
                            "pT" : (lambda j : j.pt, eqBin_pt, "p_{T} [GeV]"),
                            "eta": (lambda j : j.eta, EqBin(50 // binScaling, -2.4, 2.4), "eta"),
                            "phi": (lambda j : j.phi, EqBin(50 // binScaling, -3.1416, 3.1416), "#phi")
                    }.items() ]

        return plots