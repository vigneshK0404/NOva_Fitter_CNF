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

#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"

#include <filesystem>

const std::string min = "minimum";
const std::string optSpace = "th24vsdm41";
const std::string opt = "";
const std::string optSysts = "all";
const std::string optSamples = "numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet";
const int exp_num = 221;
const int totalBins = 148;

const std::string stem = optSpace + "_" + optSamples + "_" + std::to_string(exp_num) + "_" + optSysts;
std::string root_name = "results/" + min + "_"+ stem + ".root";
std::string file_name = min + "/" + stem + "/vals";

using namespace ana;
namespace fs = std::filesystem;

void setCalcVals_from_hist(osc::IOscCalcAdjustable* calc, TH1D* h)
{
    kFitSinSqTheta34Sterile.SetValue(calc, h->GetBinContent(1));
    kFitSinSqTheta23Sterile.SetValue(calc, std::pow(std::sin(h->GetBinContent(2)),2));
    kFitSinSqTheta24Sterile.SetValue(calc, h->GetBinContent(3));
    kFitDmSq41Sterile.SetValue(calc, std::pow(10,h->GetBinContent(4)));
    kFitDelta24InPiUnitsSterile.SetValue(calc,h->GetBinContent(5));     
    kFitDmSq32Sterile.SetValue(calc, h->GetBinContent(6));

}

void setCalcVals(osc::IOscCalcAdjustable* calc, float calcVals[])
{
    kFitDelta24InPiUnitsSterile.SetValue(calc,calcVals[0]); 
    kFitSinSqTheta24Sterile.SetValue(calc, std::pow(10,calcVals[1]));
    kFitSinSqTheta34Sterile.SetValue(calc, std::pow(10,calcVals[2]));
    kFitSinSqTheta23Sterile.SetValue(calc, calcVals[3]);
    kFitDmSq41Sterile.SetValue(calc, std::pow(10,calcVals[4]));
    kFitDmSq32Sterile.SetValue(calc,calcVals[5]);

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


 

void create_exp()
{
    TFile* f = TFile::Open((TString)root_name); 
    TH1D* h = (TH1D*)f->Get((TString)file_name);

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release(); 

    auto calc_best = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_best,"3flav");

    setCalcVals_from_hist(calc_best,h);

    nus5p1::GeneratePseudoexpt(samples, calc_best, mx, exp_num);

    auto expt = nus5p1::GetExperiment(samples, mx, opt);    
    auto multiExp = nus5p1::AddConstraints(samples, &expt, opt);
 
    std::cout << "null: " << multiExp.ChiSq(calc_best) << "\n";

    std::string outfile = "diagnoseData.root";
    auto file = ROOTFile(outfile,"recreate");
    file->cd();

    std::vector<int> expVec;
    expVec.reserve(totalBins);

    TTree* t = new TTree("dataTree","dataTree");
    t->Branch("data",&expVec);


    for(pisces::Sample& samp: samples)
    {
        Spectrum spec = samp.Data();
        Eigen::ArrayXd eigenArray = spec.GetEigen();
        expVec.insert(std::end(expVec),std::begin(eigenArray)+1,std::end(eigenArray)-1);
    }

    t->Fill();
    expVec.clear();

    t->Print();
    file->Write();
    file->Close();

    
    return;
            
}

void check_exp()
{
    
    TFile* f_true = TFile::Open((TString)root_name); 
    TH1D* h = (TH1D*)f_true->Get((TString)file_name);

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release(); 
    auto fitVars = nus5p1::GetFitVars(optSpace, opt, true);

    auto calc_true = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_true,"3flav");

    auto calc_exp = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_exp,"3flav");

    auto calc_best = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_best,"3flav");

    setCalcVals_from_hist(calc_true,h);

    nus5p1::GeneratePseudoexpt(samples, calc_true, mx, exp_num);
    auto expt = nus5p1::GetExperiment(samples, mx, opt);    
    auto multiExp = nus5p1::AddConstraints(samples, &expt, opt);
    MinuitFitter mfitter(&multiExp, fitVars, {}, MinuitFitter::kFast); 
 
    std::cout << "null: " << multiExp.ChiSq(calc_true) << "\n";

    
    TFile* f = TFile::Open("data/cnfpreds_diagnose.root");
    TTree* t = (TTree*)f->Get("tree");
    TTree* lens = (TTree*)f->Get("lens");

    Short_t len_list[1];
    lens->SetBranchAddress("lens",len_list);
    lens->GetEntry(0);

    float calcVals[6];
    t->SetBranchAddress("reps",calcVals);

    int iters = t->GetEntries();
    
    double leastChi = 1e9;

    for(int i = 0; i < iters; ++i)
    {
        t->GetEntry(i);
        setCalcVals(calc_exp,calcVals);

        double currChi = multiExp.ChiSq(calc_exp);
        printCalc(calc_exp);
        double chi2All = mfitter.Fit(calc_exp, MinuitFitter::kQuiet)->EvalMetricVal();
        printCalc(calc_exp);
        std::cout << i << " : " << currChi << " " <<chi2All << "\n";

        if (chi2All < leastChi)
        {
            leastChi = chi2All;
            setCalcVals(calc_best,calcVals);
        }        
    }

    auto calc_null_tmp = calc_true->Copy();
    double chi_null = mfitter.Fit(calc_null_tmp, MinuitFitter::kQuiet)->EvalMetricVal();

    std::cout << "\n=======================NULL HYPOTHESIS=============================\n";
    printCalc(calc_null_tmp);
    std::cout << "nullChi :" << chi_null << "\n===================================================\n";

    std::cout << "\n=======================ALTERNATE HYPOTHESIS=============================\n";
    printCalc(calc_best);
    std::cout << "leastChi : " << leastChi << "\n===================================================\n";




    return; 
}

void plotPriors()
{
    double dmsq32_min = 2.51e-3 - 6*(0.15e-3);
    double dmsq32_max = 2.51e-3 + 6*(0.15e-3);

    double params[6];

    int nBins = 30;
    TH1D* d24 = new TH1D("d24","d24",nBins,0,2);
    TH1D* ssq24 = new TH1D("ssq24","ssq24",nBins,-5,0);
    TH1D* ssq34 = new TH1D("ssq34","ssq34",nBins,-2,0);
    TH1D* ssq23 = new TH1D("ssq23","ssq23",nBins,0,1);
    TH1D* dmsq41 = new TH1D("dmsq41","dmsq41",nBins,-3,2);
    TH1D* dmsq32 = new TH1D("dmsq32","dmsq32",nBins,dmsq32_min,dmsq32_max);
    std::string path = "/pnfs/nova/scratch/users/karthiva/NOvA_CNF/data";
    for (const auto & entry : fs::directory_iterator(path))
    {
        std::string ext = entry.path().extension().string();
        if(entry.is_regular_file() && exp == ".root")
        {
            TFile f(entry);
            TTree* t = (TTree*)f.Get("Experimental_Data_Tree");

        }        
    }

}


void diagnose()
{
    //create_exp();
    //check_exp();
    plotPriors();
}



 //Generate Experiments with calc
//Ship to CNF + and get inference back
//evaluate chisq of inference points for pseudoexpt

