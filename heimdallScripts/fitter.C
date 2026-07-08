#include "CAFAna/Analysis/Calcs.h"
#include "CAFAna/Fit/MinuitFitter.h"
#include "CAFAna/Experiment/IExperiment.h"
#include "CAFAna/Experiment/MultiExperiment.h"
#include "OscLib/OscCalcSterile.h"
#include "NuXAna/Analysis/Nus5p1FitUtils.h"
#include "NuXAna/Analysis/Nus5p1PISCESHelper.h"
#include "NuXAna/Analysis/NusAna2022FitUtils.h"
#include "NuXAna/Analysis/NusAna2022PISCESHelper.h"
#include <vector>
#include "TTree.h"
#include "TFile.h"

using namespace ana;

void fitter()
{

    std::string optSpace = "th24vsdm41";
    std::string opt = "";
    std::string optSysts = "all"; //all


    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release();
    auto calc = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc,"3flav");

    nus5p1::SetData(samples,mx,1,opt);
    auto expt = nus5p1::GetExperiment(samples, mx, opt);    
    auto fitVars = nus5p1::GetFitVars(optSpace, opt, true); 
    auto multiExp = nus5p1::AddConstraints(samples,&expt,opt);

    double chiSQ = multiExp.ChiSq(calc);
    std::cout << "null: " <<chiSQ << "\n";



    std::vector<double> expVec;
    expVec.reserve(148);

    std::string outfile = "sampleData.root";
    auto file = ROOTFile(outfile,"recreate");
    TDirectory* fDir = file->mkdir("CNFData");
    file->cd();

    TTree* t = new TTree("dataTree","dataTree");
    t->Branch("data",&expVec);

    for(pisces::Sample& samp: samples)
    {
        Spectrum spec = samp.Data();
        Eigen::ArrayXd eigenArray = spec.GetEigen();
        expVec.insert(std::end(expVec),std::begin(eigenArray)+1,std::end(eigenArray)-1);
    }

    t->Fill();
    t->Print();
    t->Write();
    file->Close();



    /*TFile* f = TFile::Open("cnfpreds.root");
    TTree* t = (TTree*)f->Get("predtree");

    float calcVals[6];
    float leastVals[6] = {0,0,0,0,0,0};
    t->SetBranchAddress("preds",calcVals);

    int iters = t->GetEntries();

    double leastChi = 1e9;

    for(int i = 0; i < iters; ++i)
    {
        t->GetEntry(i);
        kFitDelta24InPiUnitsSterile.SetValue(calc,calcVals[0]); 
        kFitSinSqTheta24Sterile.SetValue(calc, calcVals[1]);
        kFitSinSqTheta34Sterile.SetValue(calc, calcVals[2]);
        kFitDmSq41Sterile.SetValue(calc, calcVals[3]);
        kFitDmSq32Sterile.SetValue(calc,calcVals[4]);
        kFitSinSqTheta23Sterile.SetValue(calc, calcVals[5]);
        
        double currChi = multiExp.ChiSq(calc);
        if (currChi < leastChi)
        {
            leastChi = currChi;
            for(int j = 0; j < 6; ++j)
            {
                leastVals[j] = calcVals[j];
            }
        }

    }

    std::cout << "Best Pred: " << leastChi << "\n";
    std::cout << "Best Vals: [";

    for(int i = 0; i < 6; ++i)
    {
        std::cout << leastVals[i] << ",";
    }
    
    std::cout << "]\n";*/



    //MinuitFitter mfitter(&multiExp, fitVars);
    //double chi2All = mfitter.Fit(calc)->EvalMetricVal();

    //std::cout << chi2All << "\n";
  
}
