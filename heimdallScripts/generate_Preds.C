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
//TODO : Amortize Random number generation
//Add Inner loop for radnom seed
//Look into kFitMethod replacement


const int totalUnique = 10000;
const int totalBins = 148;
const int totalParams = 6;
const int totalRepeat = 2048;

void generatePriors(std::vector<double>& D24, std::vector<double>& SSQ24, 
        std::vector<double>& SSQ34, std::vector<double>& DMSQ41, std::vector<double>& DMSQ32, 
        std::vector<double>& T23, std::vector<int>& SEED)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> delta_24(0, 2); //I think the function I am using contrains it with pi units so 0 to 2, it also bounds it as such.
    std::uniform_real_distribution<> sinsq_24(-4,0);
    std::uniform_real_distribution<> sinsq_34(-2,0);
    std::uniform_real_distribution<> dmsq_41(-3,2);
    std::uniform_real_distribution<> dmsq_32(-2.658,-2.552);
    std::uniform_real_distribution<> theta_23(0.5,M_PI/2);
    std::uniform_int_distribution<> NormInvOrder(0, 1);
    std::uniform_int_distribution<> RandSeed(0, totalRepeat);

    D24.reserve(totalUnique);
    SSQ24.reserve(totalUnique);
    SSQ34.reserve(totalUnique);
    DMSQ41.reserve(totalUnique);
    DMSQ32.reserve(totalUnique);
    T23.reserve(totalUnique);
    SEED.reserve(totalRepeat);

    for(int i = 0; i < totalUnique; ++i)
    {
        D24.push_back(delta_24(gen));
        SSQ24.push_back(std::pow(10,sinsq_24(gen)));
        SSQ34.push_back(std::pow(10,sinsq_34(gen)));
        T23.push_back(std::pow(10,theta_23(gen)));
        DMSQ41.push_back(std::pow(10,dmsq_41(gen)));
        double dmsq32 = std::pow(10,dmsq_32(gen));

        if(NormInvOrder(gen) == 1) 
            dmsq32 = -dmsq32;      

        DMSQ32.push_back(dmsq32);
    }

    for(int i = 0; i < totalRepeat; ++i)
    {
        SEED.push_back(RandSeed(gen));
    }
}

void generate_Preds()
{

    nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet",kPredNoSysts,true);
    auto mx = ph.GetMatrix(samples, "testCov").release();
    auto calc = nus22::GetCalc(); 

    std::vector<int> expVec;
    std::vector<double> paramVec;

    expVec.reserve(148);
    paramVec.reserve(6);

    std::vector<double> D24;
    std::vector<double> SSQ24; 
    std::vector<double> SSQ34;
    std::vector<double> DMSQ41;
    std::vector<double> DMSQ32; 
    std::vector<double> T23;
    std::vector<int> SEED;

    generatePriors(D24, SSQ24, SSQ34, DMSQ41, DMSQ32, T23, SEED);


    TFile* file = new TFile("/exp/nova/app/users/karthiva/NOvA_CNF/GeneratePreds/data/CNFData.root","RECREATE");
    file->cd();

    TTree* t = new TTree("Experimental_Data_Tree","Experimental_Data_Tree");

    t->Branch("data",&expVec);
    t->Branch("params",&paramVec);
    
    for(int i = 0; i < totalUnique; ++i)
    {        
        kFitDelta24InPiUnitsSterile.SetValue(calc,D24[i]);

        kFitSinSqTheta24Sterile.SetValue(calc,SSQ24[i]);
        kFitSinSqTheta34Sterile.SetValue(calc,SSQ34[i]);
        kFitTheta23Sterile.SetValue(calc, T23[i]);

        kFitDmSq32Sterile.SetValue(calc,DMSQ32[i]);
        kFitDmSq41Sterile.SetValue(calc,DMSQ41[i]);

        for(int j = 0; j < totalRepeat; ++j)
        {
            nus5p1::GeneratePseudoexpt(samples,calc,mx,SEED[j]);

            for(pisces::Sample& samp: samples)
            {
                Spectrum spec = samp.Data();
                Eigen::ArrayXd eigenArray = spec.GetEigen();
                expVec.insert(std::end(expVec),std::begin(eigenArray)+1,std::end(eigenArray)-1);
            }

            paramVec.push_back(D24[i]);
            paramVec.push_back(SSQ24[i]);
            paramVec.push_back(SSQ34[i]);
            paramVec.push_back(T23[i]);
            paramVec.push_back(DMSQ41[i]);
            paramVec.push_back(DMSQ32[i]);

            t->Fill();

            expVec.clear();
            paramVec.clear();
        }

        std::cout << ((float)i/totalUnique)*100 << "\% done\n";
       
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
