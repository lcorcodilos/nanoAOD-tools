import ROOT
import math, os,re, tarfile, tempfile
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.tools import matchObjectCollection, matchObjectCollectionMultiple
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetSmearer import jetSmearer
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.JetReCalibrator import JetReCalibrator

class jetmetUncertaintiesProducer(Module):
    def __init__(self, era, globalTag, jesUncertainties = [ "Total" ], jetType = "AK4PFchs", redoJEC=False, noGroom=False):

        self.era = era
        self.redoJEC = redoJEC
        self.noGroom = noGroom
        #--------------------------------------------------------------------------------------------
        # CV: globalTag and jetType not yet used in the jet smearer, as there is no consistent set of 
        #     txt files for JES uncertainties and JER scale factors and uncertainties yet
        #--------------------------------------------------------------------------------------------

        self.jesUncertainties = jesUncertainties

        # smear jet pT to account for measured difference in JER between data and simulation.
        if era == "2016":
            self.jerInputFileName = "Summer16_25nsV1_MC_PtResolution_" + jetType + ".txt"
            self.jerUncertaintyInputFileName = "Summer16_25nsV1_MC_SF_" + jetType + ".txt"
        elif era == "2017" or era == "2018": # use Fall17 as temporary placeholder until post-Moriond 2019 JERs are out
            self.jerInputFileName = "Fall17_V3_MC_PtResolution_" + jetType + ".txt"
            self.jerUncertaintyInputFileName = "Fall17_V3_MC_SF_" + jetType + ".txt"

        #jet mass resolution: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
        if self.era == "2016" or self.era == "2018": #update when 2018 values available
            self.jmrVals = [1.0, 1.2, 0.8] #nominal, up, down
        else: 
            self.jmrVals = [1.09, 1.14, 1.04]

        self.jetSmearer = jetSmearer(globalTag, jetType, self.jerInputFileName, self.jerUncertaintyInputFileName, self.jmrVals)

        if "AK4" in jetType : 
            self.jetBranchName = "Jet"
            self.genJetBranchName = "GenJet"
            self.genSubJetBranchName = None
            self.doGroomed = False
            self.corrMET = True
        elif "AK8" in jetType :
            self.jetBranchName = "FatJet"
            self.subJetBranchName = "SubJet"
            self.genJetBranchName = "GenJetAK8"
            self.genSubJetBranchName = "SubGenJetAK8"
            if not self.noGroom:
                self.doGroomed = True
                self.puppiCorrFile = ROOT.TFile.Open(os.environ['CMSSW_BASE'] + "/src/PhysicsTools/NanoAODTools/data/jme/puppiCorr.root")
                self.puppisd_corrGEN = self.puppiCorrFile.Get("puppiJECcorr_gen")
                self.puppisd_corrRECO_cen = self.puppiCorrFile.Get("puppiJECcorr_reco_0eta1v3")
                self.puppisd_corrRECO_for = self.puppiCorrFile.Get("puppiJECcorr_reco_1v3eta2v5")
            else:
                self.doGroomed = False
            self.corrMET = False
        else:
            raise ValueError("ERROR: Invalid jet type = '%s'!" % jetType)
        self.metBranchName = "MET"
        self.rhoBranchName = "fixedGridRhoFastjetAll"
        self.lenVar = "n" + self.jetBranchName

        #jet mass scale
        #W-tagging PUPPI softdrop JMS values: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
        #2016 values - use for 2018 until new values available
        self.jmsVals = [1.00, 0.9906, 1.0094] #nominal, down, up
        if self.era in ["2017","2018"]:
            self.jmsVals = [0.982, 0.978, 0.986]

        # read jet energy scale (JES) uncertainties
        # (downloaded from https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC )
        self.jesInputArchivePath = os.environ['CMSSW_BASE'] + "/src/PhysicsTools/NanoAODTools/data/jme/"
        # Text files are now tarred so must extract first into temporary directory (gets deleted during python memory management at script exit)
        self.jesArchive = tarfile.open(self.jesInputArchivePath+globalTag+".tgz", "r:gz")
        self.jesInputFilePath = tempfile.mkdtemp()
        self.jesArchive.extractall(self.jesInputFilePath)
        
        if len(jesUncertainties) == 1 and jesUncertainties[0] == "Total":
            self.jesUncertaintyInputFileName = globalTag + "_Uncertainty_" + jetType + ".txt"
        else:
            self.jesUncertaintyInputFileName = globalTag + "_UncertaintySources_" + jetType + ".txt"

        # read all uncertainty source names from the loaded file
        if jesUncertainties[0] == "All":
            with open(self.jesInputFilePath+'/'+self.jesUncertaintyInputFileName) as f:
                lines = f.read().split("\n")
                sources = filter(lambda x: x.startswith("[") and x.endswith("]"), lines)
                sources = map(lambda x: x[1:-1], sources)
                self.jesUncertainties = sources
            
        if self.redoJEC :
            self.jetReCalibrator = JetReCalibrator(globalTag, jetType , True, self.jesInputFilePath, calculateSeparateCorrections = False, calculateType1METCorrection  = False)
        

        # define energy threshold below which jets are considered as "unclustered energy"
        # (cf. JetMETCorrections/Type1MET/python/correctionTermsPfMetType1Type2_cff.py )
        self.unclEnThreshold = 15.

        # load libraries for accessing JES scale factors and uncertainties from txt files
        for library in [ "libCondFormatsJetMETObjects", "libPhysicsToolsNanoAODTools" ]:
            if library not in ROOT.gSystem.GetLibraries():
                print("Load Library '%s'" % library.replace("lib", ""))
                ROOT.gSystem.Load(library)

    def beginJob(self):

        print("Loading jet energy scale (JES) uncertainties from file '%s'" % os.path.join(self.jesInputFilePath, self.jesUncertaintyInputFileName))
        #self.jesUncertainty = ROOT.JetCorrectionUncertainty(os.path.join(self.jesInputFilePath, self.jesUncertaintyInputFileName))
    
        self.jesUncertainty = {} 
        # implementation didn't seem to work for factorized JEC, try again another way
        for jesUncertainty in self.jesUncertainties:
            jesUncertainty_label = jesUncertainty
            if jesUncertainty == 'Total' and len(self.jesUncertainties) == 1:
                jesUncertainty_label = ''
            pars = ROOT.JetCorrectorParameters(os.path.join(self.jesInputFilePath, self.jesUncertaintyInputFileName),jesUncertainty_label)
            self.jesUncertainty[jesUncertainty] = ROOT.JetCorrectionUncertainty(pars)    

        self.jetSmearer.beginJob()

    def endJob(self):
        self.jetSmearer.endJob()

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("%s_pt_raw" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_mass_raw" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_pt_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_mass_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JEC_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JER_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JMS_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JMR_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JER_up" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JMS_up" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JMR_up" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JER_down" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JMS_down" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JMR_down" % self.jetBranchName, "F", lenVar=self.lenVar)

        if self.doGroomed:
            self.out.branch("%s_msoftdrop_raw" % self.jetBranchName, "F", lenVar=self.lenVar)
            self.out.branch("%s_msoftdrop_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
            self.out.branch("%s_groomed_corr_JMR_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
            self.out.branch("%s_groomed_corr_PUPPI_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
            self.out.branch("%s_groomed_corr_JMR_up" % self.jetBranchName, "F", lenVar=self.lenVar)
            self.out.branch("%s_groomed_corr_JMR_down" % self.jetBranchName, "F", lenVar=self.lenVar)

        for jesUncertainty in self.jesUncertainties:
            self.out.branch("%s_corr_JES_%s%s" % (self.jetBranchName, jesUncertainty, shift), "F", lenVar=self.lenVar)
            # if self.doGroomed:
                # self.out.branch("%s_groomed_corr_JES_%s%s" % (self.jetBranchName, jesUncertainty, shift), "F", lenVar=self.lenVar)
            
                        
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        jets = Collection(event, self.jetBranchName )
        genJets = Collection(event, self.genJetBranchName )

        if self.doGroomed :
            subJets = Collection(event, self.subJetBranchName )
            genSubJets = Collection(event, self.genSubJetBranchName )
            genSubJetMatcher = matchObjectCollectionMultiple( genJets, genSubJets, dRmax=0.8 )

        jets_pt_raw = []
        jets_mass_raw = []
        jets_pt_nom = []
        jets_mass_nom = []

        jets_corr_JEC_nom = []
        jets_corr_JER_nom = []
        jets_corr_JMS_nom = []
        jets_corr_JMR_nom = []

        # jets_corr_JEC_up = []
        jets_corr_JES_up = {}
        jets_corr_JER_up = []
        jets_corr_JMS_up = []
        jets_corr_JMR_up = []

        # jets_corr_JEC_down = []
        jets_corr_JES_down = {}
        jets_corr_JER_down = []
        jets_corr_JMS_down = []
        jets_corr_JMR_down = []

        for jesUncertainty in self.jesUncertainties:
            jets_corr_JES_up[jesUncertainty]   = []
            jets_corr_JES_down[jesUncertainty] = []

        if self.doGroomed:
            jets_msdcorr_raw = []
            jets_msdcorr_nom = []

            jets_groomed_corr_JMR_nom = []
            jets_groomed_corr_PUPPI = []
            jets_groomed_corr_JMR_up = []
            jets_groomed_corr_JMR_down = []
            
                
        rho = getattr(event, self.rhoBranchName)

        # match reconstructed jets to generator level ones
        # (needed to evaluate JER scale factors and uncertainties)
        pairs = matchObjectCollection(jets, genJets)
 
        for jet in jets:
            genJet = pairs[jet]
            if self.doGroomed :                
                genGroomedSubJets = genSubJetMatcher[genJet] if genJet != None else None
                genGroomedJet = genGroomedSubJets[0].p4() + genGroomedSubJets[1].p4() if genGroomedSubJets != None and len(genGroomedSubJets) >= 2 else None
                if jet.subJetIdx1 >= 0 and jet.subJetIdx2 >= 0 :
                    groomedP4 = subJets[ jet.subJetIdx1 ].p4() + subJets[ jet.subJetIdx2].p4() #check subjet jecs
                else :
                    groomedP4 = None

            #jet pt and mass corrections
            jet_pt=jet.pt
            jet_mass=jet.mass

            #redo JECs if desired
            if hasattr(jet, "rawFactor"):
                jet_rawpt = jet_pt * (1 - jet.rawFactor)
                jet_rawmass = jet_mass * (1 - jet.rawFactor)
            else:
                jet_rawpt = -1.0 * jet_pt #If factor not present factor will be saved as -1
                jet_rawmass = -1.0 * jet_mass #If factor not present factor will be saved as -1
            if self.redoJEC :
                (jet_pt, jet_mass) = self.jetReCalibrator.correct(jet,rho)
            
            jets_pt_raw.append(jet_rawpt)
            jets_mass_raw.append(jet_rawmass)
            jets_pt_nom.append(jet_pt)
            jets_mass_nom.append(jet_mass)
            jets_corr_JEC_nom.append(jet_pt/jet_rawpt)

            genJet = pairs[jet]
            if self.doGroomed : 
                genGroomedSubJets = genSubJetMatcher[genJet] if genJet != None else None
                genGroomedJet = genGroomedSubJets[0].p4() + genGroomedSubJets[1].p4() if genGroomedSubJets != None and len(genGroomedSubJets) >= 2 else None
                if jet.subJetIdx1 >= 0 and jet.subJetIdx2 >= 0 :
                    groomedP4 = subJets[ jet.subJetIdx1 ].p4() + subJets[ jet.subJetIdx2].p4() #check subjet jecs
                else :
                    groomedP4 = None

                jets_msdcorr_raw.append(0.0) if groomedP4 == None else jets_msdcorr_raw.append(groomedP4.M())

                # LC: Apply PUPPI SD mass correction https://github.com/cms-jet/PuppiSoftdropMassCorr/
                puppisd_genCorr = self.puppisd_corrGEN.Eval(jet.pt)
                if abs(jet.eta) <= 1.3:
                    puppisd_recoCorr = self.puppisd_corrRECO_cen.Eval(jet.pt)
                else:
                    puppisd_recoCorr = self.puppisd_corrRECO_for.Eval(jet.pt)

                puppisd_total = puppisd_genCorr * puppisd_recoCorr
                
                if groomedP4 != None:
                    groomedP4.SetPtEtaPhiM(groomedP4.Perp(), groomedP4.Eta(), groomedP4.Phi(), groomedP4.M()*puppisd_total)

                jets_groomed_corr_PUPPI.append(puppisd_total)
                jets_msdcorr_nom.append(0.0) if groomedP4 == None else jets_msdcorr_nom.append(groomedP4.M()*puppisd_total)

            # evaluate JER scale factors and uncertainties
            # (cf. https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution and https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution )
            ( jet_corr_JER_nom, jet_corr_JER_up, jet_corr_JER_down ) = self.jetSmearer.getSmearValsPt(jet, genJet, rho)
            jets_corr_JER_nom.append(jet_corr_JER_nom)
            jets_corr_JER_up.append(jet_corr_JER_up)
            jets_corr_JER_down.append(jet_corr_JER_down)
            
            # evaluate JES uncertainties
            jet_corr_JES_up   = {}
            jet_corr_JES_down = {}
            
            # Evaluate JMS and JMR scale factors and uncertainties
            ( jet_corr_JMR_nom, jet_corr_JMR_up, jet_corr_JMR_down ) = self.jetSmearer.getSmearValsM(jet, genJet)
            jets_corr_JMS_nom.append(self.jmsVals[0])
            jets_corr_JMS_up.append(self.jmsVals[2])
            jets_corr_JMS_down.append(self.jmsVals[1])
            jets_corr_JMR_nom.append(jet_corr_JMR_nom)
            jets_corr_JMR_up.append(jet_corr_JMR_up)
            jets_corr_JMR_down.append(jet_corr_JMR_down)

            if self.doGroomed :
                # Evaluate JMR scale factors and uncertainties (JMS identical to ungroomed)
                ( jet_groomed_corr_JMR_nom, jet_groomed_corr_JMR_up, jet_groomed_corr_JMR_down ) = self.jetSmearer.getSmearValsM(groomedP4, genGroomedJet) if groomedP4 != None and genGroomedJet != None else (0.,0.,0.)
                jets_groomed_corr_JMR_nom.append(jet_groomed_corr_JMR_nom)
                jets_groomed_corr_JMR_up.append(jet_groomed_corr_JMR_up)
                jets_groomed_corr_JMR_down.append(jet_groomed_corr_JMR_down)


            
            for jesUncertainty in self.jesUncertainties:
                # (cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetCorUncertainties )
                self.jesUncertainty[jesUncertainty].setJetPt(jet.pt*jet_corr_JER_nom)
                self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                delta = self.jesUncertainty[jesUncertainty].getUncertainty(True)
                jet_corr_JES_up[jesUncertainty] = (1.+delta)
                jet_corr_JES_down[jesUncertainty] = (1.-delta)
                jets_corr_JES_up[jesUncertainty].append(jet_corr_JES_up[jesUncertainty])
                jets_corr_JES_down[jesUncertainty].append(jet_corr_JES_down[jesUncertainty])


            # propagate JER and JES corrections and uncertainties to MET
            if self.corrMET and jet.pt*jet_corr_JER_nom > self.unclEnThreshold:
                jet_cosPhi = math.cos(jet.phi)
                jet_sinPhi = math.sin(jet.phi)
                met_px_nom = met_px_nom - (jet.pt*jet_corr_JER_nom - jet_pt)*jet_cosPhi
                met_py_nom = met_py_nom - (jet.pt*jet_corr_JER_nom - jet_pt)*jet_sinPhi
                met_px_jerUp   = met_px_jerUp   - (jet.pt*jet_corr_JER_up   - jet.pt*jet_corr_JER_nom)*jet_cosPhi
                met_py_jerUp   = met_py_jerUp   - (jet.pt*jet_corr_JER_up   - jet.pt*jet_corr_JER_nom)*jet_sinPhi
                met_px_jerDown = met_px_jerDown - (jet.pt*jet_corr_JER_down - jet.pt*jet_corr_JER_nom)*jet_cosPhi
                met_py_jerDown = met_py_jerDown - (jet.pt*jet_corr_JER_down - jet.pt*jet_corr_JER_nom)*jet_sinPhi
                for jesUncertainty in self.jesUncertainties:
                    met_px_jesUp[jesUncertainty]   = met_px_jesUp[jesUncertainty]   - (jet.pt*jet_corr_JER_nom*jet_corr_JES_up[jesUncertainty]   - jet.pt*jet_corr_JER_nom)*jet_cosPhi
                    met_py_jesUp[jesUncertainty]   = met_py_jesUp[jesUncertainty]   - (jet.pt*jet_corr_JER_nom*jet_corr_JES_up[jesUncertainty]   - jet.pt*jet_corr_JER_nom)*jet_sinPhi
                    met_px_jesDown[jesUncertainty] = met_px_jesDown[jesUncertainty] - (jet.pt*jet_corr_JER_nom*jet_corr_JES_down[jesUncertainty] - jet.pt*jet_corr_JER_nom)*jet_cosPhi
                    met_py_jesDown[jesUncertainty] = met_py_jesDown[jesUncertainty] - (jet.pt*jet_corr_JER_nom*jet_corr_JES_down[jesUncertainty] - jet.pt*jet_corr_JER_nom)*jet_sinPhi

        # propagate "unclustered energy" uncertainty to MET
        if self.corrMET :
            ( met_px_unclEnUp,   met_py_unclEnUp   ) = ( met_px, met_py )
            ( met_px_unclEnDown, met_py_unclEnDown ) = ( met_px, met_py )
            met_deltaPx_unclEn = getattr(event, self.metBranchName + "_MetUnclustEnUpDeltaX")
            met_deltaPy_unclEn = getattr(event, self.metBranchName + "_MetUnclustEnUpDeltaY")
            met_px_unclEnUp    = met_px_unclEnUp   + met_deltaPx_unclEn
            met_py_unclEnUp    = met_py_unclEnUp   + met_deltaPy_unclEn
            met_px_unclEnDown  = met_px_unclEnDown - met_deltaPx_unclEn
            met_py_unclEnDown  = met_py_unclEnDown - met_deltaPy_unclEn

            # propagate effect of jet energy smearing to MET           
            met_px_jerUp   = met_px_jerUp   + (met_px_nom - met_px)
            met_py_jerUp   = met_py_jerUp   + (met_py_nom - met_py)
            met_px_jerDown = met_px_jerDown + (met_px_nom - met_px)
            met_py_jerDown = met_py_jerDown + (met_py_nom - met_py)
            for jesUncertainty in self.jesUncertainties:
                met_px_jesUp[jesUncertainty]   = met_px_jesUp[jesUncertainty]   + (met_px_nom - met_px)
                met_py_jesUp[jesUncertainty]   = met_py_jesUp[jesUncertainty]   + (met_py_nom - met_py)
                met_px_jesDown[jesUncertainty] = met_px_jesDown[jesUncertainty] + (met_px_nom - met_px)
                met_py_jesDown[jesUncertainty] = met_py_jesDown[jesUncertainty] + (met_py_nom - met_py)
            met_px_unclEnUp    = met_px_unclEnUp   + (met_px_nom - met_px)
            met_py_unclEnUp    = met_py_unclEnUp   + (met_py_nom - met_py)
            met_px_unclEnDown  = met_px_unclEnDown + (met_px_nom - met_px)
            met_py_unclEnDown  = met_py_unclEnDown + (met_py_nom - met_py)

        self.out.fillBranch("%s_pt_raw" % self.jetBranchName, jets_pt_raw)
        self.out.fillBranch("%s_pt_nom" % self.jetBranchName, jets_pt_nom)
        self.out.fillBranch("%s_mass_raw" % self.jetBranchName, jets_mass_raw)
        self.out.fillBranch("%s_mass_nom" % self.jetBranchName, jets_mass_nom)

        self.out.fillBranch("%s_corr_JEC_nom" % self.jetBranchName, jets_corr_JEC_nom)

        self.out.fillBranch("%s_corr_JER_nom" % self.jetBranchName, jets_corr_JER_nom)
        self.out.fillBranch("%s_corr_JER_up" % self.jetBranchName, jets_corr_JER_up)
        self.out.fillBranch("%s_corr_JER_down" % self.jetBranchName, jets_corr_JER_down)

        self.out.fillBranch("%s_corr_JMS_nom" % self.jetBranchName, jets_corr_JMS_nom)
        self.out.fillBranch("%s_corr_JMS_up" % self.jetBranchName, jets_corr_JMS_up)
        self.out.fillBranch("%s_corr_JMS_down" % self.jetBranchName, jets_corr_JMS_down)

        self.out.fillBranch("%s_corr_JMR_nom" % self.jetBranchName, jets_corr_JMR_nom)
        self.out.fillBranch("%s_corr_JMR_up" % self.jetBranchName, jets_corr_JMR_up)
        self.out.fillBranch("%s_corr_JMR_down" % self.jetBranchName, jets_corr_JMR_down)

                   
        if self.doGroomed :
            self.out.fillBranch("%s_groomed_corr_JMR_nom" % self.jetBranchName, jets_corr_JMR_nom)
            self.out.fillBranch("%s_groomed_corr_JMR_up" % self.jetBranchName, jets_corr_JMR_up)
            self.out.fillBranch("%s_groomed_corr_JMR_down" % self.jetBranchName, jets_corr_JMR_down)
            self.out.fillBranch("%s_groomed_corr_PUPPI" % self.jetBranchName, jets_msdcorr_corr_PUPPI)
            self.out.fillBranch("%s_msoftdrop_raw" % self.jetBranchName, jets_msdcorr_raw)
            self.out.fillBranch("%s_msoftdrop_nom" % self.jetBranchName, jets_msdcorr_nom)

            
        if self.corrMET :
            self.out.fillBranch("%s_pt_nom" % self.metBranchName, math.sqrt(met_px_nom**2 + met_py_nom**2))
            self.out.fillBranch("%s_phi_nom" % self.metBranchName, math.atan2(met_py_nom, met_px_nom))        
            self.out.fillBranch("%s_pt_jerUp" % self.metBranchName, math.sqrt(met_px_jerUp**2 + met_py_jerUp**2))
            self.out.fillBranch("%s_phi_jerUp" % self.metBranchName, math.atan2(met_py_jerUp, met_px_jerUp))        
            self.out.fillBranch("%s_pt_jerDown" % self.metBranchName, math.sqrt(met_px_jerDown**2 + met_py_jerDown**2))
            self.out.fillBranch("%s_phi_jerDown" % self.metBranchName, math.atan2(met_py_jerDown, met_px_jerDown))
            
        for jesUncertainty in self.jesUncertainties:
            self.out.fillBranch("%s_corr_JES%sUp" % (self.jetBranchName, jesUncertainty), jets_corr_JES_up[jesUncertainty])
            self.out.fillBranch("%s_corr_JES%sDown" % (self.jetBranchName, jesUncertainty), jets_corr_JES_down[jesUncertainty])

            if self.corrMET:
                self.out.fillBranch("%s_pt_jes%sUp" % (self.metBranchName, jesUncertainty), math.sqrt(met_px_jesUp[jesUncertainty]**2 + met_py_jesUp[jesUncertainty]**2))
                self.out.fillBranch("%s_phi_jes%sUp" % (self.metBranchName, jesUncertainty), math.atan2(met_py_jesUp[jesUncertainty], met_px_jesUp[jesUncertainty]))
                self.out.fillBranch("%s_pt_jes%sDown" % (self.metBranchName, jesUncertainty), math.sqrt(met_px_jesDown[jesUncertainty]**2 + met_py_jesDown[jesUncertainty]**2))
                self.out.fillBranch("%s_phi_jes%sDown" % (self.metBranchName, jesUncertainty), math.atan2(met_py_jesDown[jesUncertainty], met_px_jesDown[jesUncertainty]))
        if self.corrMET:
            self.out.fillBranch("%s_pt_unclustEnUp" % self.metBranchName, math.sqrt(met_px_unclEnUp**2 + met_py_unclEnUp**2))
            self.out.fillBranch("%s_phi_unclustEnUp" % self.metBranchName, math.atan2(met_py_unclEnUp, met_px_unclEnUp))
            self.out.fillBranch("%s_pt_unclustEnDown" % self.metBranchName, math.sqrt(met_px_unclEnDown**2 + met_py_unclEnDown**2))
            self.out.fillBranch("%s_phi_unclustEnDown" % self.metBranchName, math.atan2(met_py_unclEnDown, met_px_unclEnDown))

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
jetmetUncertainties2016 = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ])
jetmetUncertainties2016All = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "All" ])

jetmetUncertainties2017 = lambda : jetmetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "Total" ])
jetmetUncertainties2017All = lambda : jetmetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "All" ], redoJEC=True)

jetmetUncertainties2018 = lambda : jetmetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "Total" ])
jetmetUncertainties2018All = lambda : jetmetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "All" ], redoJEC=True)

jetmetUncertainties2016AK4Puppi = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ], jetType="AK4PFPuppi")
jetmetUncertainties2016AK4PuppiAll = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC",  [ "All" ], jetType="AK4PFPuppi")

jetmetUncertainties2017AK4Puppi = lambda : jetmetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "Total" ], jetType="AK4PFPuppi")
jetmetUncertainties2017AK4PuppiAll = lambda : jetmetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC",  [ "All" ], jetType="AK4PFPuppi")

jetmetUncertainties2018AK4Puppi = lambda : jetmetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "Total" ], jetType="AK4PFPuppi")
jetmetUncertainties2018AK4PuppiAll = lambda : jetmetUncertaintiesProducer("2018", "Autumn18_V8_MC",  [ "All" ], jetType="AK4PFPuppi")


jetmetUncertainties2016AK8Puppi = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ], jetType="AK8PFPuppi")
jetmetUncertainties2016AK8PuppiAll = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC",  [ "All" ], jetType="AK8PFPuppi")
jetmetUncertainties2016AK8PuppiNoGroom = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ], jetType="AK8PFPuppi",redoJEC=False,noGroom=True)
jetmetUncertainties2016AK8PuppiAllNoGroom = lambda : jetmetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", ["All"], jetType="AK8PFPuppi",redoJEC=False,noGroom=True)

jetmetUncertainties2017AK8Puppi = lambda : jetmetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "Total" ], jetType="AK8PFPuppi")
jetmetUncertainties2017AK8PuppiAll = lambda : jetmetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", ["All"], jetType="AK8PFPuppi")

jetmetUncertainties2018AK8Puppi = lambda : jetmetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "Total" ], jetType="AK8PFPuppi")
jetmetUncertainties2018AK8PuppiAll = lambda : jetmetUncertaintiesProducer("2018", "Autumn18_V8_MC", ["All"], jetType="AK8PFPuppi",redoJEC = True)
