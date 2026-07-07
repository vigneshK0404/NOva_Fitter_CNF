#include "CAFAna/Core/Utilities.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Analysis/Calcs.h"
#include "OscLib/OscCalcSterile.h"
#include "NuXAna/Analysis/Nus5p1FitUtils.h"
#include "NuXAna/Analysis/Nus5p1PISCESHelper.h"
#include <Eigen/Dense>
#include <vector>
#include "TTree.h"
#include <random>
#include <cmath>
#include "TFile.h"
#include "TString.h"

using namespace ana;

//TODO : each TRandom3 seed in Generate Pseudo expt makes slightly random expt. I need to add a method like in generateTraing data, have 1000 unique samples and a bunch of repeat ssamples, I need to think about I will store them, it may not matter as I will recombine in the end anyway
//
//TODO : Keep log scale for data
//Make gaussian for atmospheric params
//ask aurisano about fitting in log space for dmsq_41


//argc argv intial and final seeds 0->500k 500K+1 -> 1mil

const int totalUnique = 10000;
const int totalBins = 148;
const int totalParams = 6;
const int totalRepeat = 150;

void generatePriors(std::vector<double>& D24, std::vector<double>& SSQ24,
        std::vector<double>& SSQ34, std::vector<double>& DMSQ41, std::vector<double>& DMSQ32,
        std::vector<double>& SSQ23)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> delta_24(0, 2);
    std::uniform_real_distribution<> sinsq_24(-5,0);
    std::uniform_real_distribution<> sinsq_34(-2,0);
    std::uniform_real_distribution<> dmsq_41(-3,2);
    std::normal_distribution<double> dmsq_32(2.51e-3,0.15e-3);
    std::uniform_real_distribution<> sinsq_23(0,1);
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

void generate_Preds(unsigned long long int spacing)
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

    generatePriors(D24, SSQ24, SSQ34, DMSQ41, DMSQ32, SSQ23);

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


//Generate Fake -> Package them All
//ROOT FILE -> HeimDALL
//TRAIN cnf and run INFERENCE -> CALCParams (IF GOOD)
//The process
//
//GenEXP -> infer from CNF -> make regions of HPD -> list given to fitter -> runs fit
