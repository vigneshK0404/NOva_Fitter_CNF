#include "CAFAna/Core/Utilities.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Analysis/Calcs.h"
#include "CAFAna/Fit/MinuitFitter.h"
#include "CAFAna/Experiment/IExperiment.h"
#include "CAFAna/Experiment/MultiExperiment.h"
#include "OscLib/OscCalcSterile.h"
#include "NuXAna/Analysis/Nus5p1FitUtils.h"
#include "NuXAna/Analysis/Nus5p1PISCESHelper.h"
#include "NuXAna/Systs/Nus5p1Systs.h"
#include "NuXAna/Analysis/NusAna2022FitUtils.h"
#include "NuXAna/Analysis/NusAna2022PISCESHelper.h"
#include <vector>
#include "TTree.h"
#include "TFile.h"

using namespace ana;

const std::string optSpace = "th24vsdm41";
const std::string opt = "";
const std::string optSysts = "all"; //all
                                    

void createExp()
{
   
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
    file->Write();
    file->Close();   

}

void printCalc(osc::IOscCalcAdjustable* calc)
{
    std::cout << " \nDelta24(pi_units) : " <<kFitDelta24InPiUnitsSterile.GetValue(calc) << " , "
        << " ssq24 : " << kFitSinSqTheta24Sterile.GetValue(calc)  << " , "
        << " ssq34 : " << kFitSinSqTheta34Sterile.GetValue(calc) << " , "
        << " ssq23 : " << kFitSinSqTheta23Sterile.GetValue(calc) << " , "
        << " dmsq41 : " << kFitDmSq41Sterile.GetValue(calc) << " , "
        << " dmsq32 : " << kFitDmSq32Sterile.GetValue(calc) << std::endl;
    
}

void checkInference()
{
    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release();

    auto calc_null = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_null,"3flav");

    auto calc_exp = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_exp,"3flav");

    auto calc_best = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_best,"3flav");

    nus5p1::SetData(samples,mx,1,opt);
    auto expt = nus5p1::GetExperiment(samples, mx, opt);    
    auto fitVars = nus5p1::GetFitVars(optSpace, opt, true);
    auto multiExp = nus5p1::AddConstraints(samples,&expt,opt);


    double null_chiSQ = multiExp.ChiSq(calc_null);
    printCalc(calc_null);
    std::cout << "null: " << null_chiSQ << "\n";

    TFile* f = TFile::Open("data/cnfpreds.root");
    TTree* t = (TTree*)f->Get("tree");

    float calcVals[6];
    t->SetBranchAddress("reps",calcVals);

    int iters = t->GetEntries();

    double leastChi = 1e9;

    MinuitFitter mfitter(&multiExp, fitVars, {}, MinuitFitter::kFast);


    for(int i = 0; i < iters; ++i)
    {
        t->GetEntry(i); 

        kFitDelta24InPiUnitsSterile.SetValue(calc_exp,calcVals[0]); 
        kFitSinSqTheta24Sterile.SetValue(calc_exp, std::pow(10,calcVals[1]));
        kFitSinSqTheta34Sterile.SetValue(calc_exp, std::pow(10,calcVals[2]));
        kFitSinSqTheta23Sterile.SetValue(calc_exp, calcVals[3]);
        kFitDmSq41Sterile.SetValue(calc_exp, std::pow(10,calcVals[4]));
        kFitDmSq32Sterile.SetValue(calc_exp,calcVals[5]);
                
        double currChi = multiExp.ChiSq(calc_exp);
        printCalc(calc_exp);
        double chi2All = mfitter.Fit(calc_exp)->EvalMetricVal();
        std::cout << i << " : " << currChi << " " <<chi2All << "\n";

        if (chi2All < leastChi)
        {
            leastChi = chi2All;
            kFitDelta24InPiUnitsSterile.SetValue(calc_best,calcVals[0]); 
            kFitSinSqTheta24Sterile.SetValue(calc_best, std::pow(10,calcVals[1]));
            kFitSinSqTheta34Sterile.SetValue(calc_best, std::pow(10,calcVals[2]));
            kFitSinSqTheta23Sterile.SetValue(calc_best, calcVals[3]);
            kFitDmSq41Sterile.SetValue(calc_best, std::pow(10,calcVals[4]));
            kFitDmSq32Sterile.SetValue(calc_best,calcVals[5]);
        }

    }   

    std::cout << "leastChi : " << leastChi << "\n";
    printCalc(calc_best);
    
 

}

void fitter()
{
    //createExp();
    checkInference();

      
}

//MinuitFitter mfitter(&multiExp, fitVars);

