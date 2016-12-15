/*
compile with:
% javac DemoApp.java
and run with
% java DemoApp
 - must have appropriate CLASSPATH and LD_LIBRARY_PATH
 - see the libsbml java tutorial for details. 
*/

// import org.sbml.libsbml.*;
import org.sbml.jsbml.*;
import org.sbml.jsbml.xml.stax.*;

public class DemoApp
{
    public static void main(String args[])
    {
	//System.loadLibrary("sbmlj");
	String filename="/home/ndjw1/src/sbml/ch07-ar-stoch.xml";
	//String filename="ch07-ar-stoch.xml";
	SBMLReader reader = new SBMLReader();
	SBMLDocument document = reader.readSBML(filename);
	Model model = document.getModel();
	ListOf listOfSpecies = model.getListOfSpecies();
	
	for (int i = 0; i < model.getNumSpecies(); i++) {
	    Species species = (Species)listOfSpecies.get(i);
	    System.out.println(
			       species.getName() + "  " +
			       species.getCompartment() + "  " +
			       species.getInitialAmount()
			       );
	}
    }
}

