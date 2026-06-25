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
#include <random>
#include <algorithm>


using namespace ana;

//create and save randomseed vector as root file
//when making expeirments you can make them normally
//on heimdall side use awkward arrays to have the first row indicating how many rows for a particular sample
//over here save the first row and use them as indicies and pull out the saved randSeeds and use that too. 
//nohup the whole thing overnight and see what happens. 

const std::string optSpace = "th24vsdm41";
const std::string opt = "";
const std::string optSysts = "all";
const std::string optSamples = "numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet";
const int totalBins = 148;
const size_t num_randSeeds = 20;

void printCalc(osc::IOscCalcAdjustable* calc)
{
    std::cout << " \nDelta24(pi_units) : " <<kFitDelta24InPiUnitsSterile.GetValue(calc) << " , "
        << " ssq24 : " << kFitSinSqTheta24Sterile.GetValue(calc)  << " , "
        << " ssq34 : " << kFitSinSqTheta34Sterile.GetValue(calc) << " , "
        << " ssq23 : " << kFitSinSqTheta23Sterile.GetValue(calc) << " , "
        << " dmsq41 : " << kFitDmSq41Sterile.GetValue(calc) << " , "
        << " dmsq32 : " << kFitDmSq32Sterile.GetValue(calc) << std::endl;
    
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

void createExp()
{
    //creating seeds
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::uniform_int_distribution<int> dist {1, 1000};
    
    auto gen = [&](){
                   return dist(mersenne_engine);
               };

    std::vector<int> randSeeds(num_randSeeds);
    std::generate(randSeeds.begin(), randSeeds.end(), gen);

    std::string seed_outfile = "randSeeds.root";
    auto seed_file = ROOTFile(seed_outfile,"recreate");
    seed_file->cd();
    
    TTree* seed_t = new TTree("seeds","seeds");
    seed_t->Branch("seed",&randSeeds);
    seed_t->Fill();
    seed_file->Write();
    seed_file->Close();

    //Samples and CovMatrix
    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString(optSamples,kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release();

    //init calc
    auto calc_null = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_null,"3flav");

    //create root file
    std::string outfile = "sampleData.root";
    auto file = ROOTFile(outfile,"recreate");
    file->cd();

    std::vector<double> expVec;
    expVec.reserve(totalBins);

    TTree* t = new TTree("dataTree","dataTree");
    t->Branch("data",&expVec);

    
    for(const int& s : randSeeds)
    {

        nus5p1::SetData(samples,mx,s,opt);
        auto expt = nus5p1::GetExperiment(samples, mx, opt);    
        auto multiExp = nus5p1::AddConstraints(samples,&expt,opt);
 
        std::cout << "null: " << multiExp.ChiSq(calc_null) << "\n";
 
        for(pisces::Sample& samp: samples)
        {
            Spectrum spec = samp.Data();
            Eigen::ArrayXd eigenArray = spec.GetEigen();
            expVec.insert(std::end(expVec),std::begin(eigenArray)+1,std::end(eigenArray)-1);
        }


        t->Fill();
        expVec.clear();

    } 
        
    t->Print();
    file->Write();
    file->Close();   

}

void checkInference()
{
    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release();
    auto fitVars = nus5p1::GetFitVars(optSpace, opt, true);


    auto calc_null = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_null,"3flav");

    auto calc_exp = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_exp,"3flav");

    auto calc_best = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_best,"3flav");


    TFile* seed_f = TFile::Open("randSeeds.root");
    TTree* seed_t = (TTree*)seed_f->Get("seeds");
    std::vector<int>* randSeedsPtr = nullptr;
    seed_t->SetBranchAddress("seed",&randSeedsPtr);
    seed_t->GetEntry(0);
    std::vector<int> randSeeds = (*randSeedsPtr);
    seed_f->Close();


    TFile* f = TFile::Open("data/cnfpreds.root");
    TTree* t = (TTree*)f->Get("tree");
    TTree* lens = (TTree*)f->Get("lens");

    Short_t len_list[num_randSeeds];
    lens->SetBranchAddress("lens",len_list);
    lens->GetEntry(0);

    float calcVals[6];
    t->SetBranchAddress("reps",calcVals);

    int iters = t->GetEntries();
    int global_idx = 0;

 
    for(int j = 0; j < randSeeds.size(); ++j)
    {

        nus5p1::SetData(samples,mx,randSeeds[j],opt);
        auto expt = nus5p1::GetExperiment(samples, mx, opt);    
        auto multiExp = nus5p1::AddConstraints(samples,&expt,opt);
 
        double leastChi = 1e9;

        MinuitFitter mfitter(&multiExp, fitVars, {}, MinuitFitter::kFast); 
        int global_idx_cp = global_idx;

        for(int i = global_idx_cp; i < global_idx_cp + len_list[j]; ++i )
        {

            //std::cout << "i:"<< i << " global_idx_cp:" << global_idx_cp << " max:" << global_idx_cp + len_list[j] << " diff: " << len_list[j] <<"\n";
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


            global_idx++;
        }

        auto calc_null_tmp = calc_null->Copy();
        double chi_null = mfitter.Fit(calc_null_tmp, MinuitFitter::kQuiet)->EvalMetricVal();

        std::cout << "\n=======================NULL HYPOTHESIS=============================\n";
        printCalc(calc_null_tmp);
        std::cout << "nullChi :" << chi_null << "\n===================================================\n";

        std::cout << "\n=======================ALTERNATE HYPOTHESIS=============================\n";
        printCalc(calc_best);
        std::cout << "leastChi : " << leastChi << "\n===================================================\n";

        int better = 0;
        int same = 0;
        int worse = 0;

        if (leastChi < chi_null)
        {
            better++;
        }
        else if (leastChi - chi_null < 1e-3)
        {
            same++;
        }
        else
        {
            worse++;
        }

        std::cout << "Better: " << better << ", Same: " << same << ", Worse: " << worse << std::endl;

        
    }    

}

void fitter()
{
    //createExp();
    checkInference();

      
}

//MinuitFitter mfitter(&multiExp, fitVars);

