
from ternip.formats.timex3 import Timex3XmlDocument
from ternip.formats.timeml import TimeMlDocument

import ternip

# git clone https://github.com/arianpasquali/ternip
# convert from python2 to 3 if necessary (2to3 . -w)
def normalize_temporal_expressions(content, reference_date):
    """
    Constructs a corpus from documents.

    Params:
        content (str): Tokenized string
        reference_date (date): Reference date.

    """

    recogniser = ternip.recogniser() 
    normaliser = ternip.normaliser()

    content = f'<TimeML>\n{content}\n</TimeML>'
    doc = TimeMlDocument(content,"TimeML")
    sents = recogniser.tag(doc.get_sents())

    normaliser.annotate(sents, reference_date.strftime('%Y%m%d'))
    doc.reconcile(sents)

    xml_str = str(doc)

    unsupported_annotations = ["T24","T24","TMO", "TAF", "TEV", "TNI"]
    
    for ua in unsupported_annotations:
        xml_str = xml_str.replace(ua,"")

    return xml_str
