#!/usr/bin/env python
# mod2sbml.py

# Updated: 18/2/21

import libsbml,sys,re,io,traceback

__doc__="""mod2sbml version 3.1.1.1

Copyright (C) 2005-2010, Darren J Wilkinson
 d.j.wilkinson@ncl.ac.uk
 http://www.staff.ncl.ac.uk/d.j.wilkinson/

Includes modifications by:
  Jeremy Purvis (jep@thefoldingproblem.com)
  Carole Proctor (c.j.proctor@ncl.ac.uk)
  Mark Muldoon (m.muldoon@man.ac.uk)
  Lukas Endler (lukas@ebi.ac.uk)
 
This is GNU Free Software (General Public License)

Module for parsing SBML-shorthand model files, version 3.1.1,
and all previous versions

Typical usage:
>>> from mod2sbml import Parser
>>> p=Parser()
>>> p.parseStream(sys.stdin)

Raises error "ParseError" on a fatal parsing error.
"""

ParseError="Parsing error"

class Parser(object):
    """Parser class
Has constructor:
 Parser()
and the following public methods:
 parseStream(inStream)
 parse(inString)
"""
    # context
    SBML=1
    MODEL=2
    UNITS=3
    COMPARTMENTS=4
    SPECIES=5
    PARAMETERS=6
    RULES=7
    REAC1=8
    REAC2=9
    REAC3=10
    REAC4=11
    EVENTS=12

    def __init__(self):
        self.context=self.SBML
        self.count=1
        self.d=libsbml.SBMLDocument()

    def parse(self,inString):
        """parse(inString)
parses SBML-shorthand model in inString and returns a libSBML SBMLDocument
object"""
        inS=io.StringIO(inString)
        return self.parseStream(inS)

    def parseStream(self,inS):
        """parseStream(inStream)
parses SBML-shorthand model on inStream and returns a libSBML SBMLDocument
object"""
        self.inS=inS
        line=self.inS.readline()
        while (line):
            line=line.strip() # trim newline
            line=line.split("#")[0] # strip comments
            bits=line.split('"') # split off string names
            line=bits[0]
            if (len(bits)>1):
                name=bits[1]
            else:
                name=""
            line=re.sub("\s","",line) # strip whitespace
            if (line==""):
                line=self.inS.readline()
                self.count+=1
                continue # skip blank lines
            # now hand off the line to an appropriate handler
            # print self.count,line,name
            if (self.context==self.SBML):
                self.handleSbml(line,name)
            elif (self.context==self.MODEL):
                self.handleModel(line,name)
            elif (self.context==self.UNITS):
                self.handleUnits(line,name)
            elif (self.context==self.COMPARTMENTS):
                self.handleCompartments(line,name)
            elif (self.context==self.SPECIES):
                self.handleSpecies(line,name)
            elif (self.context==self.PARAMETERS):
                self.handleParameters(line,name)
            elif (self.context==self.RULES):
                self.handleRules(line,name)
            elif (self.context==self.REAC1):
                self.handleReac1(line,name)
            elif (self.context==self.REAC2):
                self.handleReac2(line,name)
            elif (self.context==self.REAC3):
                self.handleReac3(line,name)
            elif (self.context==self.REAC4):
                self.handleReac4(line,name)
            elif (self.context==self.EVENTS):
                self.handleEvents(line,name)
            line=self.inS.readline()
            self.count+=1
        self.context=self.SBML
        return self.d

    def handleSbml(self,line,name):
        # in this context, only expecting a model
        bits=line.split("=")
        morebits=bits[0].split(":")
        if ((morebits[0]!="@model")):
            sys.stderr.write('Error: expected "@model:" ')
            sys.stderr.write('at line'+str(self.count)+'\n')
            raise ParseError
        yetmorebits=morebits[1].split(".")
        level=int(yetmorebits[0])
        version=int(yetmorebits[1])
        revision=int(yetmorebits[2])
        self.mangle=100*level+10*version+revision
        if (self.mangle>311):
            sys.stderr.write('Error: shorthand version > 3.1.1 - UPGRADE CODE ')
            sys.stderr.write('at line'+str(self.count)+'\n')
            raise ParseError
        #sys.stderr.write('lev: '+str(level)+'\n') # debug
        #self.d.setLevelAndVersion(level,version)
        # HACK due to libsbml4 bug - put back when fixed...
        self.d=libsbml.SBMLDocument(level,version)
        #sys.stderr.write('Leve: '+str(self.d.getLevel())+'\n') # debug
        if (len(bits)!=2):
            sys.stderr.write('Error: expected "=" at line ')
            sys.stderr.write(str(self.count)+'\n')
            raise ParseError            
        id=bits[1]
        self.m=self.d.createModel(id)
        if (name!=""):
            self.m.setName(name)
        self.context=self.MODEL
        
    def handleModel(self,line,name):
        # in this context, expect any new context, or global units
        if (line[0]=='@'):
            self.handleNewContext(line,name)
        else:
            # hopefully a level 3 global units line...
            if (self.d.getLevel()<3):
                sys.stderr.write('Error: expected new "@section" ')
                sys.stderr.write('at line '+str(self.count)+'\n')
                raise ParseError
            # parse global units
            bits=line.split(",")
            for bit in bits:
                pair=bit.split("=")
                if len(pair)!=2:
                    sys.stderr.write('Error: expected "=" ')
                    sys.stderr.write('at line '+str(self.count)+'\n')
                    raise ParseError
                (unit,value)=pair
                if (unit=="s"):
                    self.m.setSubstanceUnits(value)
                elif (unit=="t"):
                    self.m.setTimeUnits(value)
                elif (unit=="v"):
                    self.m.setVolumeUnits(value)
                elif (unit=="a"):
                    self.m.setAreaUnits(value)
                elif (unit=="l"):
                    self.m.setLengthUnits(value)
                elif (unit=="e"):
                    self.m.setExtentUnits(value)
                elif (unit=="c"):
                    self.m.setConversionFactor(value)
                else:
                    sys.stderr.write('Error: unknown global unit ')
                    sys.stderr.write('at line '+str(self.count)+'\n')
                    raise ParseError
                    
    def handleNewContext(self,line,name):
        # sys.stderr.write('handling new context '+line[:4]+'\n')
        if (line[:4]=="@com"):
            self.context=self.COMPARTMENTS
        elif (line[:4]=="@uni"):
            self.context=self.UNITS
        elif (line[:4]=="@spe"):
            self.context=self.SPECIES
        elif (line[:4]=="@par"):
            self.context=self.PARAMETERS
        elif (line[:4]=="@rul"):
            self.context=self.RULES
        elif (line[:4]=="@rea"):
            self.context=self.REAC1
        elif (line[:4]=="@eve"):
            self.context=self.EVENTS
        else:
            sys.stderr.write('Error: unknown new "@section": '+line)
            sys.stderr.write(' at line '+str(self.count)+'\n')
            raise ParseError
        
    def handleUnits(self,line,name):
        # expect a unit or a new context
        if (line[0]=="@"):
            self.handleNewContext(line,name)
        else:
            bits=line.split("=")
            if (len(bits)<2):
                sys.stderr.write('Error: expected a "=" in: '+line)
                sys.stderr.write(' at line '+str(self.count)+'\n')
                raise ParseError
            id=bits[0]
            units="=".join(bits[1:])
            ud=self.m.createUnitDefinition()
            ud.setId(id)
            units=units.split(";")
            for unit in units:
                bits=unit.split(":")
                if (len(bits)!=2):
                    id=bits[0]
                    mods=""
                else:
                    (id,mods)=bits
                u=self.m.createUnit()
                u.setKind(libsbml.UnitKind_forName(id))
                u.setExponent(1)
                u.setMultiplier(1)
                u.setScale(0)
                mods=mods.split(",")
                for mod in mods:
                    if (mod[:2]=="e="):
                        u.setExponent(eval(mod[2:]))
                    elif (mod[:2]=="m="):
                        u.setMultiplier(eval(mod[2:]))
                    elif (mod[:2]=="s="):
                        u.setScale(eval(mod[2:]))
                    elif (mod[:2]=="o="):
                        u.setOffset(eval(mod[2:]))
            if (name!=""):
                ud.setName(name)                        

    def handleCompartments(self,line,name):
        # expect a compartment or a new context
        if (line[0]=="@"):
            self.handleNewContext(line,name)
        else:
            bits=line.split("=")
            c=bits[0]
            if (len(bits)>1):
                v=bits[1]
            else:
                v=""
            bits=c.split("<")
            com=bits[0]
            if (len(bits)>1):
                out=bits[1]
            else:
                out=""
            c=self.m.createCompartment()
            c.setId(com)
            if (out!=""):
                c.setOutside(out)
            if (v!=""):
                c.setSize(eval(v))
            if (name!=""):
                c.setName(name)
            # print self.m.toSBML()

    def handleSpecies(self,line,name):
        # expect either a species or a new section
        if (line[0]=="@"):
            self.handleNewContext(line,name)
        else:
            bits=line.split("=")
            if (len(bits)!=2):
                sys.stderr.write('Error: expected "=" on line ')
                sys.stderr.write(str(self.count)+'\n')
                raise ParseError
            (bit,amount)=bits
            bits=bit.split(":")
            if (len(bits)!=2):
                sys.stderr.write('Error: expected ":" on line ')
                sys.stderr.write(str(self.count)+'\n')
                raise ParseError            
            (comp,id)=bits
            if (id[0]=="[" and id[-1]=="]"):
                conc=True
                id=id[1:-1]
            else:
                conc=False
            s=self.m.createSpecies()
            s.setId(id)
            s.setCompartment(comp)
            split=re.search('[a-df-z]',amount)
            if (split!=None):
                split=split.start()
                opts=amount[split:]
                amount=amount[:split]
            else:
                opts=""
            while (opts!=""):
                if (opts[0]=="b"):
                    s.setBoundaryCondition(True)
                elif (opts[0]=="c"):
                    s.setConstant(True)
                elif (opts[0]=="s"):
                    s.setHasOnlySubstanceUnits(True)
                opts=opts[1:]
            if (conc):
                s.setInitialConcentration(eval(amount))
            else:
                s.setInitialAmount(eval(amount))
            if (name!=""):
                s.setName(name)
            #print self.d.toSBML()

    def handleParameters(self,line,name):
        # expect either a parameter or a new section
        if (line[0]=="@"):
            self.handleNewContext(line,name)
        else:
            bits=line.split("=")
            p=self.m.createParameter()
            p.setId(bits[0])
            if (len(bits)!=2):
                sys.stderr.write('Error: expected "=" on line ')
                sys.stderr.write(str(self.count)+'\n')
                raise ParseError
            (bit,value)=bits
            split=re.search('[a-df-z]',value)
            if (split!=None):
                split=split.start()
                opts=value[split:]
                value=value[:split]
            else:
                opts=""
            while (opts!=""):
                if (opts[0]=="v"):
                    p.setConstant(False)
                opts=opts[1:]
            p.setValue(eval(value))    
            if (name!=""):
                p.setName(name)
            #print self.d.toSBML()

    def handleRules(self,line,name):
        # expect either a rule or a new section
        # changed by lukas: rules are AssignmentRule by default
        # "@assign:" and "@rate:" let choose between different kinds of rules
        # this requires the assigned species to have atrribute
        # constant set to "False"
        if (line[0]=="@" and (not re.match("@(?:rate|assign)\:",line))):
            self.handleNewContext(line,name)
        else:
            rule_type=1 # standard set to assignment rule (1)
            if re.match("@rate\:",line):
                rule_type=2 # set to rate rule
            elif re.match("@assign\:",line):
                rule_type=1 # set to assignment rule
            line=re.sub("\@\w+\:","",line) # replace rule declaration if there
            bits=line.split("=")
            if (len(bits)!=2):
                sys.stderr.write('Error: expected "=" on line ')
                sys.stderr.write(str(self.count)+'\n')
                raise ParseError
            (lhs,rhs)=bits
            value=libsbml.parseFormula(rhs)
            self.replaceTime(value)
            if rule_type==1:
                rule=self.m.createAssignmentRule()
            elif rule_type==2:
                rule=self.m.createRateRule()
            rule.setVariable(lhs)
            rule.setMath(value)
            # print self.d.toSBML()
            
    def handleReac1(self,line,name):
        # expect a reaction or a new context
        if (line[:3]!="@r=" and line[:4]!="@rr="):
            self.handleNewContext(line,name)
        else:
            bits=line.split("=")
            if (len(bits)!=2):
                sys.stderr.write('Error: expected "=" on line ')
                sys.stderr.write(str(self.count)+'\n')
                raise ParseError
            (tag,id)=bits
            if (tag!="@r" and tag!="@rr"):
                sys.stderr.write('Error: expected "@r=" on line ')
                sys.stderr.write(str(self.count)+'\n')
                raise ParseError
            self.r=self.m.createReaction()
            self.r.setId(id)
            if (tag=="@r"):
                self.r.setReversible(False)
            else:
                self.r.setReversible(True)
            if (name!=""):
                self.r.setName(name)
            self.context=self.REAC2

    def handleReac2(self,line,name):
        # expect a reaction equation and possibly modifiers
        # of form:
        # A + B -> C : M1, M2, M3
        chks=line.split(":")
        if (len(chks)>1):
            pars=chks[1].split(",")
            for par in pars:
                mdf = self.r.createModifier()
                mdf.setSpecies(par)
        bits=chks[0].split("->")
        if (len(bits)!=2):
            sys.stderr.write('Error: expected "->" on line ')
            sys.stderr.write(str(self.count)+'\n')
            raise ParseError
        (lhs,rhs)=bits
        if (lhs):
            self.handleTerms(lhs,True)
        if (rhs):
            self.handleTerms(rhs,False)
        self.context=self.REAC3

    def handleTerms(self,side,left):
        terms=side.split("+")
        for term in terms:
            #split=re.search('\D',term).start()
            split=re.search('[^\d\.]',term).start()
            if (split==0):
                sto=1.0
            else:
                sto=eval(term[:split])
            id=term[split:]
            if (left):
                sr=self.r.createReactant()
            else:
                sr=self.r.createProduct()
            sr.setSpecies(id)
            sr.setStoichiometry(sto)

    def handleReac3(self,line,name):
        # expect a kinetic law, a new reaction or a new context
        if (line[:3]=="@r=" or line[:4]=="@rr="):
            self.handleReac1(line,name)
        elif (line[0]=="@"):
            self.handleNewContext(line,name)
        else:
            bits=line.split(":")
            form=bits[0]
            kl=self.r.createKineticLaw()
            kl.setFormula(form)
            if (len(bits)>1):
                pars=bits[1].split(",")
                for par in pars:
                    bits=par.split("=")
                    if (len(bits)!=2):
                        sys.stderr.write('Error: expected "=" on ')
                        sys.stderr.write('line '+str(self.count)+'\n')
                        raise ParseError
                    (id,value)=bits
                    parm=kl.createParameter()
                    parm.setId(id)
                    parm.setValue(eval(value))
            self.context=self.REAC1

    def handleEvents(self,line,name):
        # expect an event, or a new context
        if (line[0]=="@"):
            self.handleNewContext(line,name)
        else:
            bits=line.split(":")
            if (len(bits)!=2):
                sys.stderr.write('Error: expected exactly one ":" on ')
                sys.stderr.write('line '+str(self.count)+'\n')
                raise ParseError
            (event,assignments)=bits
            bits=event.split(";")
            trigbits=bits[0].split("=")
            if (len(trigbits)<2):
                sys.stderr.write('Error: expected a "=" before ":" on ')
                sys.stderr.write('line '+str(self.count)+'\n')
                raise ParseError
            id=trigbits[0]
            trig="=".join(trigbits[1:])
            e=self.m.createEvent()
            e.setId(id)
            trig=self.trigMangle(trig)
            triggerMath=libsbml.parseFormula(trig)
            self.replaceTime(triggerMath)
            trigger=e.createTrigger()
            trigger.setMath(triggerMath)
            if (len(bits)==2):
                delay=e.createDelay()
                delayMath=libsbml.parseFormula(bits[1])
                delay.setMath(delayMath)
	    # SPLIT
            if (self.mangle>=230):
              	asslist=assignments.split(";")
            else:
                asslist=assignments.split(",")
            for ass in asslist:
                bits=ass.split("=")
                if (len(bits)!=2):
                    sys.stderr.write('Error: expected exactly one "=" in assignment on')
                    sys.stderr.write('line '+str(self.count)+'\n')
                    raise ParseError
                (var,math)=bits
                ea=self.m.createEventAssignment()
                ea.setVariable(var)
                ea.setMath(libsbml.parseFormula(math))
            if (name!=""):
                e.setName(name)
                
    def trigMangle(self,trig):
        bits=trig.split(">=")
        if (len(bits)==2):
            return self.binaryOp("geq",bits)
        bits=trig.split("<=")
        if (len(bits)==2):
            return self.binaryOp("leq",bits)
        bits=trig.split(">")
        if (len(bits)==2):
            return self.binaryOp("gt",bits)
        bits=trig.split("<")
        if (len(bits)==2):
            return self.binaryOp("lt",bits)
        bits=trig.split("=")
        if (len(bits)==2):
            return self.binaryOp("eq",bits)
        return trig
        
    def binaryOp(self,op,bits):
        return(op+"("+bits[0]+","+bits[1]+")")

    def replaceTime(self,ast):
        if (ast.getType()==libsbml.AST_NAME):
            if ((ast.getName()=='t') or (ast.getName()=='time')):
                ast.setType(libsbml.AST_NAME_TIME)
        for node in range(ast.getNumChildren()):
            self.replaceTime(ast.getChild(node))


# if run as a script...

if __name__=='__main__':
    p=Parser()
    argc=len(sys.argv)
    try:
        if (argc==1):
            d=p.parseStream(sys.stdin)
        else:
            try:
                s=open(sys.argv[1],"r")
            except:
                sys.stderr.write('Error: failed to open file: ')
                sys.stderr.write(sys.argv[1]+'\n')
                sys.exit(1)
            d=p.parseStream(s)
        print('<?xml version="1.0" encoding="UTF-8"?>')
        print(d.toSBML())
    except:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write('\n\n Unknown parsing error!\n')
        sys.exit(1)
    
