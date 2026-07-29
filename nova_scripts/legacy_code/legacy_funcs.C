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

void find_basin(double calc_means[], double calc_stds[])
{
    nus5p1::SetData(samples, mx, exp_num, opt);
    auto expt = nus5p1::GetExperiment(samples, mx, opt);    
    auto multiExp = nus5p1::AddConstraints(samples,&expt,opt);

    MinuitFitter mfitter(&multiExp, fitVars, {}, MinuitFitter::kFast); 

    double chi_best = multiExp.ChiSq(calc_best);

    std::cout << "Best Chi : " << chi_best << "\n";
    
    double ratio = 0.01; 

    
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

            plus_condition  = (chi_diff_plus <= 10);
            minus_condition  = (chi_diff_minus <= 10);

                        
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

void plotPriors() //UNFINISHED
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

