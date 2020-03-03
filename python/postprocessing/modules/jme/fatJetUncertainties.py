import ROOT
import math, os,re, tarfile, tempfile, shutil
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.tools import matchObjectCollection, matchObjectCollectionMultiple
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetSmearer import jetSmearer
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.JetReCalibrator import JetReCalibrator

class fatJetUncertaintiesProducer(Module):
    def __init__(self, era, globalTag, jesUncertainties = [ "Total" ], archive=None, jetType = "AK8PFPuppi", redoJEC=False, noGroom=False, jerTag="", jmrVals = [], jmsVals = [], isData=False, applySmearing=True):

        self.era = era
        self.redoJEC = redoJEC
        self.noGroom = noGroom
        self.isData = isData
        self.applySmearing = applySmearing if not isData else False # don't smear for data
        #--------------------------------------------------------------------------------------------
        # CV: globalTag and jetType not yet used in the jet smearer, as there is no consistent set of 
        #     txt files for JES uncertainties and JER scale factors and uncertainties yet
        #--------------------------------------------------------------------------------------------

        self.jesUncertainties = jesUncertainties
        # smear jet pT to account for measured difference in JER between data and simulation.
        if jerTag != "":
            self.jerInputFileName = jerTag + "_PtResolution_" + jetType + ".txt"
            self.jerUncertaintyInputFileName = jerTag + "_SF_"  + jetType + ".txt"
        else:
            print "WARNING: jerTag is empty!!! This module will soon be deprecated! Please use jetmetHelperRun2 in the future."
            if era == "2016":
                self.jerInputFileName = "Summer16_25nsV1_MC_PtResolution_" + jetType + ".txt"
                self.jerUncertaintyInputFileName = "Summer16_25nsV1_MC_SF_" + jetType + ".txt"
            elif era == "2017" or era == "2018": # use Fall17 as temporary placeholder until post-Moriond 2019 JERs are out
                self.jerInputFileName = "Fall17_V3_MC_PtResolution_" + jetType + ".txt"
                self.jerUncertaintyInputFileName = "Fall17_V3_MC_SF_" + jetType + ".txt"

        #jet mass resolution: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
        self.jmrVals = jmrVals
        if not self.jmrVals:
            print "WARNING: jmrVals is empty!!! Using default values. This module will soon be deprecated! Please use jetmetHelperRun2 in the future."
            self.jmrVals = [1.0, 1.2, 0.8] #nominal, up, down
            # Use 2017 values for 2018 until 2018 are released
            if self.era in ["2017","2018"]:
                self.jmrVals = [1.09, 1.14, 1.04] 

        self.jetSmearer = jetSmearer(globalTag, jetType, self.jerInputFileName, self.jerUncertaintyInputFileName, self.jmrVals)

        if "AK4" in jetType : 
            raise ValueError('ERROR: Jet type cannot be AK4 for fatJetUncertaintiesProducer')
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
        else:
            raise ValueError("ERROR: Invalid jet type = '%s'!" % jetType)
        self.rhoBranchName = "fixedGridRhoFastjetAll"
        self.lenVar = "n" + self.jetBranchName

        #jet mass scale
        self.jmsVals = jmsVals
        if not self.jmsVals:
            print "WARNING: jmsVals is empty!!! Using default values! This module will soon be deprecated! Please use jetmetHelperRun2 in the future."
            #2016 values 
            self.jmsVals = [1.00, 1.0094, 0.9906] #nominal, up, down
            # Use 2017 values for 2018 until 2018 are released
            if self.era in ["2017","2018"]:
                self.jmsVals = [0.982, 0.986, 0.978]

        # read jet energy scale (JES) uncertainties
        # (downloaded from https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC )
        self.jesInputArchivePath = os.environ['CMSSW_BASE'] + "/src/PhysicsTools/NanoAODTools/data/jme/"
        # Text files are now tarred so must extract first into temporary directory (gets deleted during python memory management at script exit)
        self.jesArchive = tarfile.open(self.jesInputArchivePath+globalTag+".tgz", "r:gz") if not archive else tarfile.open(self.jesInputArchivePath+archive+".tgz", "r:gz")
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
        shutil.rmtree(self.jesInputFilePath)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("%s_pt_raw" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_pt_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_mass_raw" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_mass_nom" % self.jetBranchName, "F", lenVar=self.lenVar)
        self.out.branch("%s_corr_JEC" % self.jetBranchName, "F", lenVar=self.lenVar)
        
        if not self.isData:
            for shift in ["", "_up", "_down" ]:
                self.out.branch("%s_corr_JER%s" % (self.jetBranchName, shift), "F", lenVar=self.lenVar)
                self.out.branch("%s_corr_JMS%s" % (self.jetBranchName, shift), "F", lenVar=self.lenVar)
                self.out.branch("%s_corr_JMR%s" % (self.jetBranchName, shift), "F", lenVar=self.lenVar)
                if shift != "":
                    for jesUncertainty in self.jesUncertainties:
                            self.out.branch("%s_corr_JES%s%s" % (self.jetBranchName, jesUncertainty, shift), "F", lenVar=self.lenVar)

        if self.doGroomed:
            self.out.branch("%s_msoftdrop_raw" % self.jetBranchName, "F", lenVar=self.lenVar)
            self.out.branch("%s_msoftdrop_nom" % self.jetBranchName, "F", lenVar=self.lenVar)            
            self.out.branch("%s_msoftdrop_corr_PUPPI" % self.jetBranchName, "F", lenVar=self.lenVar)

            if not self.isData:    
                for shift in ["", "_up", "_down" ]:
                    self.out.branch("%s_msoftdrop_corr_JMR%s" % (self.jetBranchName, shift), "F", lenVar=self.lenVar)
                    self.out.branch("%s_msoftdrop_tau21DDT_corr_JMR%s" % (self.jetBranchName, shift), "F", lenVar=self.lenVar)
                    self.out.branch("%s_msoftdrop_tau21DDT_corr_JMS%s" % (self.jetBranchName, shift), "F", lenVar=self.lenVar)


    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        jets = Collection(event, self.jetBranchName )
        if not self.isData:
          genJets = Collection(event, self.genJetBranchName )
        
        if self.doGroomed :
            subJets = Collection(event, self.subJetBranchName )
            if not self.isData:
              genSubJets = Collection(event, self.genSubJetBranchName )
              genSubJetMatcher = matchObjectCollectionMultiple( genJets, genSubJets, dRmax=0.8 )
        
        self.jetSmearer.setSeed(event)
        
        jets_pt_raw = []
        jets_pt_nom = []
        jets_mass_raw = []
        jets_mass_nom = []
        jets_corr_JEC = []
        
        if not self.isData:
            jets_corr_JER = []
            jets_corr_JMS = []
            jets_corr_JMR = []
            jets_corr_JER_up = []
            jets_corr_JMS_up = []
            jets_corr_JMR_up = []
            jets_corr_JER_down = []
            jets_corr_JMS_down = []
            jets_corr_JMR_down = []
        
            jets_corr_JES_up   = {}
            jets_corr_JES_down = {}
            for jesUncertainty in self.jesUncertainties:
                jets_corr_JES_up[jesUncertainty]   = []
                jets_corr_JES_down[jesUncertainty] = []
        
        if self.doGroomed:
            jets_msdcorr_raw = []
            jets_msdcorr_nom = []
            jets_msdcorr_corr_PUPPI = []

            if not self.isData:
                jets_msdcorr_corr_JMR   = []
                jets_msdcorr_corr_JMR_up   = []
                jets_msdcorr_corr_JMR_down   = []
            
                jets_msdcorr_tau21DDT_corr_JMR = []
                jets_msdcorr_tau21DDT_corr_JMS = []
                jets_msdcorr_tau21DDT_corr_JMR_up = []
                jets_msdcorr_tau21DDT_corr_JMS_up = []
                jets_msdcorr_tau21DDT_corr_JMR_down = []
                jets_msdcorr_tau21DDT_corr_JMS_down = []

        rho = getattr(event, self.rhoBranchName)
        
        # match reconstructed jets to generator level ones
        # (needed to evaluate JER scale factors and uncertainties)
        if not self.isData:
            pairs = matchObjectCollection(jets, genJets)
        
        for jet in jets:
            ###########################################
            # JEC undo and redo and JES uncertainties #
            ###########################################
            #jet pt and mass 
            jet_pt=jet.pt
            jet_mass=jet.mass
            
            # Redo JECs if desired
            if hasattr(jet, "rawFactor"):
                jet_rawpt = jet_pt * (1 - jet.rawFactor)
                jet_rawmass = jet_mass * (1 - jet.rawFactor)
            else:
                jet_rawpt = -1.0 * jet_pt #If factor not present factor will be saved as -1
                jet_rawmass = -1.0 * jet_mass #If factor not present factor will be saved as -1
            if self.redoJEC :
                (jet_pt, jet_mass) = self.jetReCalibrator.correct(jet,rho)
                jet.pt = jet_pt
                jet.mass = jet_mass
            # Store raw (without JECs) and nominal (with JECs)
            jets_pt_raw.append(jet_rawpt)
            jets_mass_raw.append(jet_rawmass)
            jets_pt_nom.append(jet_pt)
            jets_mass_nom.append(jet_mass)
            jets_corr_JEC.append(jet_pt/jet_rawpt)

            #############################
            # Groomed JEC undo and redo #
            #############################
            if self.doGroomed:
                # Loop over soft drop subjets (only 2 by definition)
                if jet.subJetIdx1 >= 0 and jet.subJetIdx2 >= 0 :
                    # Need to undo JECs of subjets to get raw mass
                    groomedP4_raw = ROOT.Math.PtEtaPhiMVector(0,0,0,0)
                    groomedP4_nom = ROOT.Math.PtEtaPhiMVector(0,0,0,0)
                    for isj in [jet.subJetIdx1,jet.subJetIdx2]:
                        sj = subJets[isj]
                        if hasattr(sj, "rawFactor"):
                            sj_rawpt = sj.pt * (1 - sj.rawFactor)
                            sj_rawmass = sj.mass * (1 - sj.rawFactor)
                        else: 
                            sj_rawpt = -1.0 * sj.pt
                            sj_rawmass = -1.0 * sj.mass

                        sj_raw_vect = ROOT.Math.PtEtaPhiMVector(sj_rawpt,sj.eta,sj.phi,sj_rawmass)
                        sj_nom_vect = ROOT.Math.PtEtaPhiMVector(sj.pt,sj.eta,sj.phi,sj.mass)

                    groomedP4_raw = groomedP4_raw + sj_raw_vect 
                    groomedP4_nom = groomedP4_nom + sj_nom_vect 
                else :
                    groomedP4_raw = None
                    groomedP4_nom = None

                # Store raw and nominal masses
                jets_msdcorr_raw.append(groomedP4_raw.M() if groomedP4_raw != None else 0.0) #raw value always stored without JECs
                jets_msdcorr_nom.append(groomedP4_nom.M() if groomedP4_nom != None else 0.0) #nom value always stored with JECs

                #############################################
                # Lookup PUPPI SD mass correction and store #
                #############################################
                # https://github.com/cms-jet/PuppiSoftdropMassCorr/
                # Should be applied by analyzer to raw msoftdrop
                puppisd_genCorr = self.puppisd_corrGEN.Eval(jet.pt)
                if abs(jet.eta) <= 1.3:
                    puppisd_recoCorr = self.puppisd_corrRECO_cen.Eval(jet.pt)
                else:
                    puppisd_recoCorr = self.puppisd_corrRECO_for.Eval(jet.pt)
                puppisd_total = puppisd_genCorr * puppisd_recoCorr
                jets_msdcorr_corr_PUPPI.append(puppisd_total)


            #################################################################
            # Do other corrections, smearing, and uncertainties if not data #
            #################################################################
            if not self.isData:
                genJet = pairs[jet] 
                #######
                # JER #
                #######
                # Evaluate JER scale factors and uncertainties
                # (cf. https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution and https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution )
                ( jet_corr_jerNomVal, jet_corr_jerUpVal, jet_corr_jerDownVal ) = self.jetSmearer.getSmearValsPt(jet, genJet, rho)
                # Store corrections and variations
                jets_corr_JER.append(jet_corr_jerNomVal)
                jets_corr_JER_up.append(jet_corr_jerUpVal)
                jets_corr_JER_down.append(jet_corr_jerDownVal)

                #######
                # JES #
                #######
                for jesUncertainty in self.jesUncertainties:
                    # (cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetCorUncertainties )
                    self.jesUncertainty[jesUncertainty].setJetPt(jet_pt*jet_corr_jerNomVal)
                    self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                    delta = self.jesUncertainty[jesUncertainty].getUncertainty(True)
                    jets_corr_JES_up[jesUncertainty].append(1+delta)
                    jets_corr_JES_down[jesUncertainty].append(1-delta) 
                
                #######
                # JMS #
                #######
                # Grab JMR scale factors and uncertainties
                jet_corr_jmsNomVal, jet_corr_jmsUpVal, jet_corr_jmsDownVal = self.jmsVals
                # Store corrections and variations
                jets_corr_JMS.append(jet_corr_jmsNomVal)
                jets_corr_JMS_up.append(jet_corr_jmsUpVal)
                jets_corr_JMS_down.append(jet_corr_jmsDownVal)

                #######
                # JMR #
                #######
                # Evaluate JMR scale factors and uncertainties
                ( jet_corr_jmrNomVal, jet_corr_jmrUpVal, jet_corr_jmrDownVal ) = self.jetSmearer.getSmearValsM(jet, genJet)
                # Store corrections and variations
                jets_corr_JMR.append(jet_corr_jmrNomVal)
                jets_corr_JMR_up.append(jet_corr_jmrUpVal)
                jets_corr_JMR_down.append(jet_corr_jmrDownVal)

                ##################################
                # Groomed variations of JMR, JMS #
                ##################################
                if self.doGroomed :
                    # Get matched gen jets if MC 
                    genGroomedSubJets = genSubJetMatcher[genJet] if genJet != None else None
                    genGroomedJet = genGroomedSubJets[0].p4() + genGroomedSubJets[1].p4() if genGroomedSubJets != None and len(genGroomedSubJets) >= 2 else None

                    #######
                    # JMS #
                    #######
                    # Same as for non-groomed jet so skip
                    # Note that it is applied to the PUPPI SD mass corrected mass (ie. raw * PUPPI SD mass correction * jms)

                    #######
                    # JMR #
                    #######
                    # Evaluate JMR scale factors and uncertainties
                    groomedP4_corr = groomedP4_raw # Eventually has PUPPI SD mass correction and JMS correction for JMR calculation
                    if groomedP4_corr != None: groomedP4_corr.SetM(groomedP4_corr.M()*puppisd_total*jet_corr_jmsNomVal)
                    ( jet_msdcorr_jmrNomVal, jet_msdcorr_jmrUpVal, jet_msdcorr_jmrDownVal ) = self.jetSmearer.getSmearValsM(groomedP4_corr, genGroomedJet) if (groomedP4_corr != None and genGroomedJet != None) else (1.,1.,1.)

                    jets_msdcorr_corr_JMR.append(jet_msdcorr_jmrNomVal)
                    jets_msdcorr_corr_JMR_up.append(jet_msdcorr_jmrUpVal)
                    jets_msdcorr_corr_JMR_down.append(jet_msdcorr_jmrDownVal)

                    #Also evaluated JMS&JMR SD corr in tau21DDT region: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetWtagging#tau21DDT_0_43
                    if self.era in ["2016"]:
                        jets_msdcorr_tau21DDT_jmsNomVal = 1.014
                        jets_msdcorr_tau21DDT_jmsDownVal = 1.007
                        jets_msdcorr_tau21DDT_jmsUpVal = 1.021
                        self.jetSmearer.jmr_vals = [1.086,1.176,0.996]
                    elif self.era in ["2017"]:
                        jets_msdcorr_tau21DDT_jmsNomVal = 0.983
                        jets_msdcorr_tau21DDT_jmsDownVal = 0.976
                        jets_msdcorr_tau21DDT_jmsUpVal = 0.99
                        self.jetSmearer.jmr_vals = [1.080,1.161,0.999]
                    elif self.era in ["2018"]:
                        jets_msdcorr_tau21DDT_jmsNomVal = 1.000   # tau21DDT < 0.43 WP
                        jets_msdcorr_tau21DDT_jmsDownVal = 0.990
                        jets_msdcorr_tau21DDT_jmsUpVal = 1.010
                        self.jetSmearer.jmr_vals = [1.124,1.208,1.040]

                    ( jet_msdcorr_tau21DDT_jmrNomVal, jet_msdcorr_tau21DDT_jmrUpVal, jet_msdcorr_tau21DDT_jmrDownVal ) = self.jetSmearer.getSmearValsM(groomedP4_corr, genGroomedJet) if groomedP4_corr != None and genGroomedJet != None else (0.,0.,0.)

                    jets_msdcorr_tau21DDT_corr_JMR.append(jet_msdcorr_tau21DDT_jmrNomVal)
                    jets_msdcorr_tau21DDT_corr_JMR_up.append(jet_msdcorr_tau21DDT_jmrUpVal)
                    jets_msdcorr_tau21DDT_corr_JMR_down.append(jet_msdcorr_tau21DDT_jmrDownVal)
                    jets_msdcorr_tau21DDT_corr_JMS.append(jets_msdcorr_tau21DDT_jmsNomVal)
                    jets_msdcorr_tau21DDT_corr_JMS_up.append(jets_msdcorr_tau21DDT_jmsUpVal)
                    jets_msdcorr_tau21DDT_corr_JMS_down.append(jets_msdcorr_tau21DDT_jmsDownVal)

                    #Restore original jmr_vals in jetSmearer
                    self.jetSmearer.jmr_vals = self.jmrVals
                                   

        self.out.fillBranch("%s_pt_raw" % self.jetBranchName, jets_pt_raw)
        self.out.fillBranch("%s_pt_nom" % self.jetBranchName, jets_pt_nom)
        self.out.fillBranch("%s_corr_JEC" % self.jetBranchName, jets_corr_JEC)
        self.out.fillBranch("%s_mass_raw" % self.jetBranchName, jets_mass_raw)
        self.out.fillBranch("%s_mass_nom" % self.jetBranchName, jets_mass_nom)

        if not self.isData:
            self.out.fillBranch("%s_corr_JER" % self.jetBranchName, jets_corr_JER)
            self.out.fillBranch("%s_corr_JMS" % self.jetBranchName, jets_corr_JMS)
            self.out.fillBranch("%s_corr_JMR" % self.jetBranchName, jets_corr_JMR)
            self.out.fillBranch("%s_corr_JER_up" % self.jetBranchName, jets_corr_JER_up)
            self.out.fillBranch("%s_corr_JMS_up" % self.jetBranchName, jets_corr_JMS_up)
            self.out.fillBranch("%s_corr_JMR_up" % self.jetBranchName, jets_corr_JMR_up)
            self.out.fillBranch("%s_corr_JER_down" % self.jetBranchName, jets_corr_JER_down)
            self.out.fillBranch("%s_corr_JMS_down" % self.jetBranchName, jets_corr_JMS_down)
            self.out.fillBranch("%s_corr_JMR_down" % self.jetBranchName, jets_corr_JMR_down)
            for jesUncertainty in self.jesUncertainties:
                self.out.fillBranch("%s_corr_JES%s_up" % (self.jetBranchName, jesUncertainty), jets_corr_JES_up[jesUncertainty])
                self.out.fillBranch("%s_corr_JES%s_down" % (self.jetBranchName, jesUncertainty), jets_corr_JES_down[jesUncertainty])
            
        if self.doGroomed :
            self.out.fillBranch("%s_msoftdrop_raw" % self.jetBranchName, jets_msdcorr_raw)
            self.out.fillBranch("%s_msoftdrop_nom" % self.jetBranchName, jets_msdcorr_nom)
            self.out.fillBranch("%s_msoftdrop_corr_PUPPI" % self.jetBranchName, jets_msdcorr_corr_PUPPI)
            
            if not self.isData:
                self.out.fillBranch("%s_msoftdrop_corr_JMR" % self.jetBranchName, jets_msdcorr_corr_JMR)
                self.out.fillBranch("%s_msoftdrop_corr_JMR_up" % self.jetBranchName, jets_msdcorr_corr_JMR_up)
                self.out.fillBranch("%s_msoftdrop_corr_JMR_down" % self.jetBranchName, jets_msdcorr_corr_JMR_down)
                self.out.fillBranch("%s_msoftdrop_tau21DDT_corr_JMR" % self.jetBranchName, jets_msdcorr_tau21DDT_corr_JMR)
                self.out.fillBranch("%s_msoftdrop_tau21DDT_corr_JMR_up" % self.jetBranchName, jets_msdcorr_tau21DDT_corr_JMR_up)
                self.out.fillBranch("%s_msoftdrop_tau21DDT_corr_JMR_down" % self.jetBranchName, jets_msdcorr_tau21DDT_corr_JMR_down)
                self.out.fillBranch("%s_msoftdrop_tau21DDT_corr_JMS" % self.jetBranchName, jets_msdcorr_tau21DDT_corr_JMS)
                self.out.fillBranch("%s_msoftdrop_tau21DDT_corr_JMS_up" % self.jetBranchName, jets_msdcorr_tau21DDT_corr_JMS_up)
                self.out.fillBranch("%s_msoftdrop_tau21DDT_corr_JMS_down" % self.jetBranchName, jets_msdcorr_tau21DDT_corr_JMS_down)
                    
        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
fatJetUncertainties2016 = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ])
fatJetUncertainties2016All = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "All" ])

fatJetUncertainties2017 = lambda : fatJetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "Total" ])
fatJetUncertainties2017All = lambda : fatJetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "All" ], redoJEC=True)

fatJetUncertainties2018 = lambda : fatJetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "Total" ])
fatJetUncertainties2018All = lambda : fatJetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "All" ], redoJEC=True)

fatJetUncertainties2016AK4Puppi = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ], jetType="AK4PFPuppi")
fatJetUncertainties2016AK4PuppiAll = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC",  [ "All" ], jetType="AK4PFPuppi")

fatJetUncertainties2017AK4Puppi = lambda : fatJetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "Total" ], jetType="AK4PFPuppi")
fatJetUncertainties2017AK4PuppiAll = lambda : fatJetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC",  [ "All" ], jetType="AK4PFPuppi")

fatJetUncertainties2018AK4Puppi = lambda : fatJetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "Total" ], jetType="AK4PFPuppi")
fatJetUncertainties2018AK4PuppiAll = lambda : fatJetUncertaintiesProducer("2018", "Autumn18_V8_MC",  [ "All" ], jetType="AK4PFPuppi")


fatJetUncertainties2016AK8Puppi = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ], jetType="AK8PFPuppi")
fatJetUncertainties2016AK8PuppiAll = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC",  [ "All" ], jetType="AK8PFPuppi")
fatJetUncertainties2016AK8PuppiNoGroom = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", [ "Total" ], jetType="AK8PFPuppi",redoJEC=False,noGroom=True)
fatJetUncertainties2016AK8PuppiAllNoGroom = lambda : fatJetUncertaintiesProducer("2016", "Summer16_07Aug2017_V11_MC", ["All"], jetType="AK8PFPuppi",redoJEC=False,noGroom=True)

fatJetUncertainties2017AK8Puppi = lambda : fatJetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", [ "Total" ], jetType="AK8PFPuppi")
fatJetUncertainties2017AK8PuppiAll = lambda : fatJetUncertaintiesProducer("2017", "Fall17_17Nov2017_V32_MC", ["All"], jetType="AK8PFPuppi")

fatJetUncertainties2018AK8Puppi = lambda : fatJetUncertaintiesProducer("2018", "Autumn18_V8_MC", [ "Total" ], jetType="AK8PFPuppi")
fatJetUncertainties2018AK8PuppiAll = lambda : fatJetUncertaintiesProducer("2018", "Autumn18_V8_MC", ["All"], jetType="AK8PFPuppi",redoJEC = True)

