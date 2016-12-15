/*
DemoApp.java

Simple demo of jSBML
 
*/

import org.sbml.jsbml.*;

public class DemoApp
{

    public static void main(String args[]) throws javax.xml.stream.XMLStreamException, java.io.IOException
    {
	String filename="ch07-mm-stoch.xml";
	SBMLReader reader = new SBMLReader();
	SBMLDocument document = reader.readSBML(filename);
	Model model = document.getModel();
	ListOf listOfSpecies = model.getListOfSpecies();
	
	for (int i = 0; i < model.getNumSpecies(); i++) {
	    Species species = (Species)listOfSpecies.get(i);
	    System.out.println(
			       species.getId() + "\t" +
			       species.getName() + "\t" +
			       species.getCompartment() + "\t" +
			       species.getInitialAmount()
			       );
	}
    }

}


/* eof */


