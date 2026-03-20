#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/Utilities.h"
#include "CAFAna/Prediction/IPrediction.h"
#include "PISCES/Experiment/LikelihoodCovMxExperiment.h"
#include "CAFAna/Analysis/Calcs.h"

#include "OscLib/OscCalcSterile.h"

#include "TRandom3.h"

#include "NuXAna/Systs/Nus5p1Systs.h"
#include "NuXAna/Analysis/Nus5p1FitUtils.h"
#include "NuXAna/Analysis/Nus5p1PISCESHelper.h"

#include <sstream>
using namespace ana;

void GeneratePseudoexpt(std::vector<pisces::Sample>& samples,
                            osc::IOscCalcSterile* calc,
                            pisces::CovarianceMatrix* mx,
                            int seed)
    {
      // get stat seeds
      TRandom3 rng(seed);
      std::map<pisces::Sample, int> statSeeds;
      for (pisces::Sample& s : pisces::Sample::All()) statSeeds[s] = rng.Integer(1e9);

      // get syst shifts
      std::vector<Spectrum> shifted;
      if (mx) shifted = mx->ThrowPseudoexpt(samples, calc, seed);
      else for (const pisces::Sample& s : samples) {
        Spectrum spec = s.Predict(calc).AsimovData(s.POT());
        spec.OverrideLivetime(s.Livetime());
        shifted.push_back(spec);
      } // for sample

      // add cosmics and Poisson fluctuate
      for (size_t iS = 0; iS < samples.size(); ++iS) {
        pisces::Sample& s = samples[iS];
        if (s.HasCosmic()) shifted[iS] += s.Cosmic();
        s.SetData(shifted[iS].MockData(s.POT(), statSeeds[s]));
      } // for sample
    } // function GeneratePseudoexpt

void generate_Preds()
{

    osc::IOscCalcSterile* calc = new osc::OscCalcSterile();
    /*Setting Up the Calc*/
    calc->SetL(1.0);
    calc->SetRho(1.0);
    calc->SetAngle(4,1,1.0);
    calc->SetDelta(4,1,1.0);
    calc->SetDm(4,1.0);

    ana::nus5p1::PISCESHelper ph;
    auto samples = ph.GetSamplesFromOptString("SampleString", kPredNoSysts, true); 
    auto mx = ph.GetMatrix(samples, "SystString").release();
    nus5p1::SetData(samples, mx, "DataString", "OptString");

    GeneratePseudoexpt(samples,calc,mx,10);
}
