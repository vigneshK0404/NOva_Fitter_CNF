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
#include <random>
#include <cmath>

#include <filesystem>

const std::string min = "minimum";
const std::string optSpace = "th24vsdm41";
const std::string opt = "";
const std::string optSysts = "all";
const std::string optSamples = "numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet";
const int totalBins = 148;

const int totalUnique = 10000;
const int totalParams = 6;
const int totalRepeat = 150;

using namespace ana;
namespace fs = std::filesystem;

double norm_angle(double angle, double div) {
    double result = std::fmod(angle, div);
    if (result < 0) {
        result += div;
    }
    return result;
}

void setCalcVals_from_hist(osc::IOscCalcAdjustable* calc, TH1D* h)
{
    kFitSinSqTheta34Sterile.SetValue(calc, h->GetBinContent(1));
    kFitTheta23Sterile.SetValue(calc, h->GetBinContent(2));
    kFitSinSqTheta24Sterile.SetValue(calc, h->GetBinContent(3));
    kFitDmSq41Sterile.SetValue(calc, std::pow(10,h->GetBinContent(4)));
    kFitDelta24InPiUnitsSterile.SetValue(calc,h->GetBinContent(5));     
    kFitDmSq32Sterile.SetValue(calc, h->GetBinContent(6));

}

std::vector<double> printCalc(osc::IOscCalcAdjustable* calc)
{
    std::vector<double> calcVals = {kFitDelta24InPiUnitsSterile.GetValue(calc),kFitTheta24Sterile.GetValue(calc),
        kFitTheta34Sterile.GetValue(calc),kFitTheta23Sterile.GetValue(calc),
        kFitDmSq41Sterile.GetValue(calc),kFitDmSq32Sterile.GetValue(calc)};

    std::cout << " \nDelta24(pi_units) : " << calcVals[0] << " , "
        << " ssq24 : " << calcVals[1]  << " , "
        << " ssq34 : " << calcVals[2] << " , "
        << " t23 : " << calcVals[3] << " , "
        << " dmsq41 : " << calcVals[4] << " , "
        << " dmsq32 : " << calcVals[5] << std::endl;

    return calcVals;
    
}




void generatePriors_around_basin(std::vector<double>& D24, std::vector<double>& SSQ24,
        std::vector<double>& SSQ34, std::vector<double>& DMSQ41, std::vector<double>& DMSQ32,
        std::vector<double>& SSQ23, int exp_num)
{
    const std::string stem = optSpace + "_" + optSamples + "_" + std::to_string(exp_num) + "_" + optSysts;
    std::string root_name = "results/" + min + "_"+ stem + ".root";
    std::string file_name = min + "/" + stem + "/vals";

    TFile* f = TFile::Open((TString)root_name); 
    TH1D* h = (TH1D*)f->Get((TString)file_name);

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString(optSamples,kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release(); 
    auto fitVars = nus5p1::GetFitVars(optSpace, opt, true);

    auto calc_best = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_best,"3flav");

    setCalcVals_from_hist(calc_best,h);
    
    std::vector<double> calcVals = printCalc(calc_best);

    calc_means[0] = calcVals[0]; 
    calc_means[1] = std::log10(calcVals[1]);
    calc_means[2] = std::log10(calcVals[2]);
    calc_means[3] = calcVals[3]; 
    calc_means[4] = std::log10(calcVals[4]);
    calc_means[5] = calcVals[5];

    calc_stds[0] = calcVals[0]/4; 
    calc_stds[1] = std::log10(calcVals[1])*0.1;
    calc_stds[2] = std::log10(calcVals[2])*0.1;
    calc_stds[3] = calcVals[3]/4; 
    calc_stds[4] = std::log10(calcVals[4])*0.1;
    calc_stds[5] = calcVals[5]/4;


    std::cout << "PRINTING Bounds\n";
    for(int k = 0; k < 6; ++k)
    {
        double plus = calc_means[k] + 3*calc_stds[k];
        double minus = calc_means[k] - 3*calc_stds[k];
        std::cout << "Param: " << k << " (" << minus << " - " << plus << ")\n";
    }


    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double>  delta_24(calc_means[0], calc_stds[0]);
    std::normal_distribution<double>  sinsq_24(calc_means[1], calc_stds[1]);
    std::normal_distribution<double>  sinsq_34(calc_means[2], calc_stds[2]);
    std::normal_distribution<double>  dmsq_41(calc_means[3], calc_stds[3]);
    std::normal_distribution<double> dmsq_32(calc_means[4],0.12e-3);
    std::normal_distribution<double>  sinsq_23(calc_means[5], calc_stds[5]);
    std::uniform_int_distribution<> NormInvOrder(0, 1);


    D24.reserve(totalUnique);
    SSQ24.reserve(totalUnique);
    SSQ34.reserve(totalUnique);
    DMSQ41.reserve(totalUnique);
    DMSQ32.reserve(totalUnique);
    SSQ23.reserve(totalUnique);

    for(int i = 0; i < totalUnique; ++i)
    {
        D24.push_back(delta_24(gen));
        SSQ24.push_back(sinsq_24(gen));
        SSQ34.push_back(sinsq_34(gen));
        SSQ23.push_back(sinsq_23(gen));
        DMSQ41.push_back(dmsq_41(gen));
        double dmsq32 = dmsq_32(gen);

        if(NormInvOrder(gen) == 1)
            dmsq32 = -dmsq32;

        DMSQ32.push_back(dmsq32);
    }

}




void generate_Preds(unsigned long long int spacing, int exp_num)
{

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString(optSamples,kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, "all").release();
    auto calc = nus22::GetCalc();

    std::vector<int> expVec;
    std::vector<double> paramVec;

    expVec.reserve(totalBins);
    paramVec.reserve(totalParams);

    std::vector<double> D24;
    std::vector<double> SSQ24;
    std::vector<double> SSQ34;
    std::vector<double> DMSQ41;
    std::vector<double> DMSQ32;
    std::vector<double> SSQ23;

    generatePriors_around_basin(D24, SSQ24, SSQ34, DMSQ41, DMSQ32, SSQ23, exp_num);

    int idx = RunningOnGrid() ? JobNumber() : 0;
    unsigned long long int minSeed = (unsigned long long int)idx*spacing;
    unsigned long long int maxSeed = minSeed + spacing;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long> RandSeed(minSeed, maxSeed);

    std::string outfile = "CNFData_"+std::to_string(minSeed)+"_"+std::to_string(maxSeed)+ "_" + std::to_string(idx) + ".root";
    auto file = ROOTFile(outfile,"recreate");
    TDirectory* fDir = file->mkdir("CNFData");
    file->cd();

    TTree* t = new TTree("Experimental_Data_Tree","Experimental_Data_Tree");

    t->Branch("data",&expVec);
    t->Branch("params",&paramVec);


    for(int i = 0; i < totalUnique; ++i)
    {
        kFitDelta24InPiUnitsSterile.SetValue(calc,D24[i]);

        kFitSinSqTheta24Sterile.SetValue(calc,std::pow(10,SSQ24[i]));
        kFitSinSqTheta34Sterile.SetValue(calc,std::pow(10,SSQ34[i]));
        kFitSinSqTheta23Sterile.SetValue(calc, SSQ23[i]);

        kFitDmSq32Sterile.SetValue(calc,DMSQ32[i]);
        kFitDmSq41Sterile.SetValue(calc,std::pow(10,DMSQ41[i]));

        for(int j = 0; j < totalRepeat; ++j)
        {
            nus5p1::GeneratePseudoexpt(samples,calc,mx,RandSeed(gen));

            for(pisces::Sample& samp: samples)
            {
                Spectrum spec = samp.Data();
                Eigen::ArrayXd eigenArray = spec.GetEigen();
                expVec.insert(std::end(expVec),std::begin(eigenArray)+1,std::end(eigenArray)-1);
            }

            paramVec.push_back(D24[i]);
            paramVec.push_back(SSQ24[i]);
            paramVec.push_back(SSQ34[i]);
            paramVec.push_back(SSQ23[i]);
            paramVec.push_back(DMSQ41[i]);
            paramVec.push_back(DMSQ32[i]);

            t->Fill();

            expVec.clear();
            paramVec.clear();
        }

        if((i % (totalUnique/100)) == 0)
            std::cout << ((float)i * 100/totalUnique) << "\% done\n";

    }

    t->Print();
    t->Write();
    file->Close();

}



void diagnose(int exp_num)
{
    generate_Preds(1000000,exp_num);
}



 //Generate Experiments with calc
//Ship to CNF + and get inference back
//evaluate chisq of inference points for pseudoexpt

