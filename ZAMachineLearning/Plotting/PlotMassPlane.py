import os
import sys
import argparse
import copy
import numpy as np
import ROOT

sys.path.append(os.path.abspath('..'))
from talos import Restore
from preprocessing import PreprocessLayer
from tdrstyle import setTDRStyle

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

class MassPlane:
    def __init__(self,x_bins,x_min,x_max,y_bins,y_min,y_max,plot_DY=False,plot_TT=False,plot_ZA=False):
        self.x_bins     = x_bins
        self.x_min      = x_min
        self.x_max      = x_max
        self.y_bins     = y_bins
        self.y_min      = y_min
        self.y_max      = y_max
        self.model      = None
        self.plot_DY    = plot_DY
        self.plot_TT    = plot_TT
        self.plot_ZA    = plot_ZA
        self.graph_list = []

        # Produce grid #
        self.produce_grid()


    def produce_grid(self):
        self.X,self.Y = np.meshgrid(np.linspace(self.x_min,self.x_max,self.x_bins),np.linspace(self.y_min,self.y_max,self.y_bins))
        bool_upper = np.greater_equal(self.Y,self.X)
        self.X = self.X[bool_upper]
        self.Y = self.Y[bool_upper]
        self.x = self.X.reshape(-1,1)
        self.y = self.Y.reshape(-1,1)
        # X, Y are 2D arrays, x,y are vectors of points

    def load_model(self,path_model):
        self.model = Restore(path_model, custom_objects={'PreprocessLayer': PreprocessLayer}).model
        self.model_name = os.path.basename(path_model).replace(".zip","")


    def plotMassPoint(self,mH,mA):
        print ("Producing plot for MH = %d GeV, MA = %d"%(mH,mA))
        N = self.x.shape[0]
        params = np.c_[np.ones(N)*mA,np.ones(N)*mH]
        inputs = np.c_[self.x,self.y,params]
        output = self.model.predict(inputs)

        g_DY = ROOT.TGraph2D(N)
        g_DY.SetNpx(500)
        g_DY.SetNpy(500)
        g_TT = ROOT.TGraph2D(N)
        g_TT.SetNpx(500)
        g_TT.SetNpy(500)
        g_ZA = ROOT.TGraph2D(N)
        g_ZA.SetNpx(500)
        g_ZA.SetNpy(500)

        for i in range(N):
            if self.plot_DY:
                g_DY.SetPoint(i,self.x[i],self.y[i],output[i,0])
            if self.plot_TT:
                g_TT.SetPoint(i,self.x[i],self.y[i],output[i,1])
            if self.plot_ZA:
                g_ZA.SetPoint(i,self.x[i],self.y[i],output[i,2])

        if self.plot_DY:
            self.graph_list.append(g_DY)
            g_DY.GetHistogram().SetTitle("P(DY) for mass point M_{H} = %d GeV, M_{A} = %d GeV"%(mH,mA))
            g_DY.GetHistogram().GetXaxis().SetTitle("M_{jj} [GeV]")
            g_DY.GetHistogram().GetYaxis().SetTitle("M_{lljj} [GeV]")
            g_DY.GetHistogram().GetZaxis().SetTitle("DNN output")
            g_DY.GetHistogram().GetZaxis().SetRangeUser(0.,1.)
            g_DY.GetHistogram().SetContour(100)
            g_DY.GetXaxis().SetTitleOffset(1.2)
            g_DY.GetYaxis().SetTitleOffset(1.2)
            g_DY.GetZaxis().SetTitleOffset(1.2)
            g_DY.GetXaxis().SetTitleSize(0.045)
            g_DY.GetYaxis().SetTitleSize(0.045)
            g_DY.GetZaxis().SetTitleSize(0.045)

        if self.plot_TT:
            g_TT.GetHistogram().SetTitle("P(t#bar{t}) for mass point M_{H} = %d GeV, M_{A} = %d GeV"%(mH,mA))
            g_TT.GetHistogram().GetXaxis().SetTitle("M_{jj} [GeV]")
            g_TT.GetHistogram().GetYaxis().SetTitle("M_{lljj} [GeV]")
            g_TT.GetHistogram().GetZaxis().SetTitle("DNN output")
            g_TT.GetHistogram().GetZaxis().SetRangeUser(0.,1.)
            g_TT.GetHistogram().SetContour(100)
            g_TT.GetXaxis().SetTitleOffset(1.2)
            g_TT.GetYaxis().SetTitleOffset(1.2)
            g_TT.GetZaxis().SetTitleOffset(1.2)
            g_TT.GetXaxis().SetTitleSize(0.045)
            g_TT.GetYaxis().SetTitleSize(0.045)
            g_TT.GetZaxis().SetTitleSize(0.045)
            self.graph_list.append(g_TT)

        if self.plot_ZA:
            g_ZA.GetHistogram().SetTitle("P(H#rightarrowZA) for mass point M_{H} = %d GeV, M_{A} = %d GeV"%(mH,mA))
            g_ZA.GetHistogram().GetXaxis().SetTitle("M_{jj} [GeV]")
            g_ZA.GetHistogram().GetYaxis().SetTitle("M_{lljj} [GeV]")
            g_ZA.GetHistogram().GetZaxis().SetTitle("DNN output")
            g_ZA.GetHistogram().GetZaxis().SetRangeUser(0.,1.)
            g_ZA.GetHistogram().SetContour(100)
            g_ZA.GetXaxis().SetTitleOffset(1.2)
            g_ZA.GetYaxis().SetTitleOffset(1.2)
            g_ZA.GetZaxis().SetTitleOffset(1.2)
            g_ZA.GetXaxis().SetTitleSize(0.045)
            g_ZA.GetYaxis().SetTitleSize(0.045)
            g_ZA.GetZaxis().SetTitleSize(0.045)
            self.graph_list.append(g_ZA)


    def plotOnCanvas(self):
        setTDRStyle()
        pdf_path = "MassPlane/"+self.model_name+".pdf"
        C = ROOT.TCanvas("C","C",800,600)
        #C.SetLogz()
        C.Print(pdf_path+"[")
        for g in self.graph_list:
            print ("Plotting %s"%g.GetTitle())
            g.Draw("colz")
            g_copy = g.Clone()
            contours = np.array([0.90,0.95,0.99])
            g_copy.GetHistogram().SetContour(contours.shape[0],contours)
            g_copy.Draw("cont2 same")
            C.Print(pdf_path,"Title:"+g.GetTitle())
        C.Print(pdf_path+"]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to plot mass plane with Neural Net')
    parser.add_argument('--model', action='store', required=True, type=str, 
                   help='Path to model (in zip file)')
    parser.add_argument('-mA', action='store', required=False, type=int, 
                   help='Mass of A for plot')
    parser.add_argument('-mH', action='store', required=False, type=int, 
                   help='Mass of H for plot')
    parser.add_argument('-DY', action='store_true', required=False, default=False,
                   help='Wether to plot the DY output')
    parser.add_argument('-TT', action='store_true', required=False, default=False,
                   help='Wether to plot the TT output')
    parser.add_argument('-ZA', action='store_true', required=False, default=False,
                   help='Wether to plot the ZA output')
    parser.add_argument('--gif', action='store_true', required=False, default=False, 
                   help='Wether to produce the gif on all mass plane (overriden by --mA and --mH)')
    args = parser.parse_args()

    inst = MassPlane(200,0,1500,200,0,1500,args.DY,args.TT,args.ZA)
    inst.load_model(args.model)


    if args.mA and args.mH:
        inst.plotMassPoint(args.mH,args.mA)
    elif not args.gif:
        inst.plotMassPoint(200,50)
        inst.plotMassPoint(200,100)
        inst.plotMassPoint(250,50)
        inst.plotMassPoint(250,100)
        inst.plotMassPoint(300,50)
        inst.plotMassPoint(300,100)
        inst.plotMassPoint(300,200)
        inst.plotMassPoint(500,50)
        inst.plotMassPoint(500,100)
        inst.plotMassPoint(500,200)
        inst.plotMassPoint(500,300)
        inst.plotMassPoint(500,400)
        inst.plotMassPoint(650,50)
        inst.plotMassPoint(800,50)
        inst.plotMassPoint(800,100)
        inst.plotMassPoint(800,200)
        inst.plotMassPoint(800,400)
        inst.plotMassPoint(800,700)
        inst.plotMassPoint(1000,50)
        inst.plotMassPoint(1000,100)
        inst.plotMassPoint(1000,200)
        inst.plotMassPoint(1000,500)

    inst.plotOnCanvas()
