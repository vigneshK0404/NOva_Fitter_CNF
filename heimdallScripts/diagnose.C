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
const int exp_num = 1;
const int totalBins = 148;

const int totalUnique = 10000;
const int totalParams = 6;
const int totalRepeat = 150;


const std::string stem = optSpace + "_" + optSamples + "_" + std::to_string(exp_num) + "_" + optSysts;
std::string root_name = "results/" + min + "_"+ stem + ".root";
std::string file_name = min + "/" + stem + "/vals";

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

void setCalcVals_from_list(osc::IOscCalcAdjustable* calc, std::vector<double> calcVals)
{
    kFitDelta24InPiUnitsSterile.SetValue(calc,calcVals[0]); 
    kFitTheta24Sterile.SetValue(calc, calcVals[1]);
    kFitTheta34Sterile.SetValue(calc, calcVals[2]);
    kFitTheta23Sterile.SetValue(calc, calcVals[3]);
    kFitDmSq41Sterile.SetValue(calc, calcVals[4]);
    kFitDmSq32Sterile.SetValue(calc,calcVals[5]);

}

void setCalcVals_from_CNF(osc::IOscCalcAdjustable* calc, float calcVals[])
{
    kFitDelta24InPiUnitsSterile.SetValue(calc,calcVals[0]); 
    kFitSinSqTheta24Sterile.SetValue(calc, std::pow(10,calcVals[1]));
    kFitSinSqTheta34Sterile.SetValue(calc, std::pow(10,calcVals[2]));
    kFitSinSqTheta23Sterile.SetValue(calc, calcVals[3]);
    kFitDmSq41Sterile.SetValue(calc, std::pow(10,calcVals[4]));
    kFitDmSq32Sterile.SetValue(calc,calcVals[5]);

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

std::vector<double> getFinalCalc(std::vector<double> calcVals_tmp)
{
    auto calc = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc,"3flav");
    
    setCalcVals_from_list(calc, calcVals_tmp);

    std::vector<double> calcVals = {kFitDelta24InPiUnitsSterile.GetValue(calc),kFitSinSqTheta24Sterile.GetValue(calc),
        kFitSinSqTheta34Sterile.GetValue(calc),kFitSinSqTheta23Sterile.GetValue(calc),
        kFitDmSq41Sterile.GetValue(calc),kFitDmSq32Sterile.GetValue(calc)};

    return calcVals;

}

void find_basin(double calc_means[], double calc_stds[])
{
    TFile* f = TFile::Open((TString)root_name); 
    TH1D* h = (TH1D*)f->Get((TString)file_name);

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, optSysts).release(); 
    auto fitVars = nus5p1::GetFitVars(optSpace, opt, true);

    auto calc_best = nus5p1::GetOscCalcForFitting(optSpace, opt);
    nus22::SetParams(calc_best,"3flav");

    setCalcVals_from_hist(calc_best,h);
    
    std::vector<double> calcVals = printCalc(calc_best);

    nus5p1::SetData(samples, mx, exp_num, opt);
    auto expt = nus5p1::GetExperiment(samples, mx, opt);    
    auto multiExp = nus5p1::AddConstraints(samples,&expt,opt);

    MinuitFitter mfitter(&multiExp, fitVars, {}, MinuitFitter::kFast); 

    double chi_best = multiExp.ChiSq(calc_best);

    std::cout << "Best Chi : " << chi_best << "\n";
    
    double ratio = 0.15; 

    
    std::vector<double> calcVals_plus_tmp(6);
    std::vector<double> calcVals_minus_tmp(6);


    for(int param_idx = 0; param_idx < 6; ++param_idx)
    {
        
        auto calc_plus = calc_best->Copy();
        auto calc_minus = calc_best->Copy();

        double param = calcVals[param_idx];
        double delta = ratio * param; 

        for(int i = 0; i < 6; ++i)
        {
            calcVals_plus_tmp[i] = calcVals[i];
            calcVals_minus_tmp[i] = calcVals[i];
        }

        bool plus_condition = true;
        bool minus_condition = true;
        
        int patience = 0;

        std::cout << "Param idx: " << param_idx << std::endl;

        do
        {   

            double chi_plus = multiExp.ChiSq(calc_plus);
            double chi_minus = multiExp.ChiSq(calc_minus);

            double chi_diff_plus = std::abs(chi_plus - chi_best);
            double chi_diff_minus = std::abs(chi_minus - chi_best);

            plus_condition  = (chi_diff_plus <= 3);
            minus_condition  = (chi_diff_minus <= 3);

                        
            double num = 0;

            if(plus_condition)
            {
                num = calcVals_plus_tmp[param_idx];
                num += delta;

                switch(param_idx)
                {
                    case 0:
                        num = norm_angle(num,2.0);
                    case 1:
                        num = norm_angle(num,2*M_PI);
                    case 2:
                        num = norm_angle(num,2*M_PI);
                    case 3:
                        num = norm_angle(num,2*M_PI);
                        break;
                }

                calcVals_plus_tmp[param_idx] = num;

                setCalcVals_from_list(calc_plus,calcVals_plus_tmp); 
            }


            if(minus_condition)
            {
                num = calcVals_minus_tmp[param_idx];
                num -= delta;
                
                switch(param_idx)
                {
                    case 0:
                        num = norm_angle(num,2.0);
                    case 1:
                        num = norm_angle(num,2*M_PI);
                    case 2:
                        num = norm_angle(num,2*M_PI);
                    case 3:
                        num = norm_angle(num,2*M_PI);
                        break;
                }

                calcVals_minus_tmp[param_idx] = num;


                setCalcVals_from_list(calc_minus,calcVals_minus_tmp);
            }


            if((multiExp.ChiSq(calc_plus) - chi_plus) <= 1e-5 && (multiExp.ChiSq(calc_minus) - chi_minus) <= 1e-5)
                patience++;

            std::cout << calcVals_plus_tmp[param_idx] << " " << calcVals_minus_tmp[param_idx] << " " <<chi_diff_plus << " "<< chi_diff_minus <<"\n";

        
        }while((plus_condition || minus_condition) && patience < 20);
        
    }

    std::vector<double> calcVals_plus = getFinalCalc(calcVals_plus_tmp); 
    std::vector<double> calcVals_minus = getFinalCalc(calcVals_minus_tmp);


    calc_means[0] = 0.5*(calcVals_plus[0] + calcVals_minus[0]); 
    calc_stds[0] = 0.5*(calcVals_plus[0] - calcVals_minus[0]);

    calc_means[1] = 0.5*(std::log10(calcVals_plus[1]) + std::log10(calcVals_minus[1])); 
    calc_stds[1] = 0.5*(std::log10(calcVals_plus[1]) - std::log10(calcVals_minus[1]));

    calc_means[2] = 0.5*(std::log10(calcVals_plus[2]) + std::log10(calcVals_minus[2])); 
    calc_stds[2] = 0.5*(std::log10(calcVals_plus[2]) - std::log10(calcVals_minus[2]));

    calc_means[3] = 0.5*(calcVals_plus[3] + calcVals_minus[3]); 
    calc_stds[3] = 0.5*(calcVals_plus[3] - calcVals_minus[3]);

    calc_means[4] = 0.5*(std::log10(calcVals_plus[4]) + std::log10(calcVals_minus[4])); 
    calc_stds[4] = 0.5*(std::log10(calcVals_plus[4]) - std::log10(calcVals_minus[4]));

    calc_means[5] = 0.5*(calcVals_plus[5] + calcVals_minus[5]); 
    calc_stds[5] = 0.5*(calcVals_plus[5] - calcVals_minus[5]);




}

void generatePriors_around_basin(std::vector<double>& D24, std::vector<double>& SSQ24,
        std::vector<double>& SSQ34, std::vector<double>& DMSQ41, std::vector<double>& DMSQ32,
        std::vector<double>& SSQ23, double calc_means[], double calc_stds[])
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double>  delta_24(calc_means[0], calc_stds[0]);
    std::normal_distribution<double>  sinsq_24(calc_means[1], calc_stds[1]);
    std::normal_distribution<double>  sinsq_34(calc_means[2], calc_stds[2]);
    std::normal_distribution<double>  dmsq_41(calc_means[3], calc_stds[3]);
    std::normal_distribution<double> dmsq_32(calc_means[4], calc_stds[4]);
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




void generate_Preds(unsigned long long int spacing, double calc_means[], double calc_stds[])
{

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
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

    generatePriors_around_basin(D24, SSQ24, SSQ34, DMSQ41, DMSQ32, SSQ23,calc_means,calc_stds);

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

            //std::cout << i << j <<"\n";

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


/*  
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
    return; 
}
*/

void diagnose()
{
    double calc_means[6];
    double calc_stds[6];

    find_basin(calc_means, calc_stds);


    for(int i = 0; i < 6; ++i)
    {
        std::cout << "param: " << i << " Value: " << calc_means[i] << " Std: " << calc_stds[i] << "\n"; 
    }
    generate_Preds(1000000,calc_means,calc_stds);

}



 //Generate Experiments with calc
//Ship to CNF + and get inference back
//evaluate chisq of inference points for pseudoexpt

