import xml.dom.minidom
for i in range(3):
    # Open XML file
    try:
        dom = xml.dom.minidom.parse(
            "static/texts/84951c943faa2db40140ee004db22a25-1.2.xml")
        root = dom.documentElement
        cc = dom.getElementsByTagName('text')
        c1 = cc[0]
        print(c1.firstChild.data)
        text = "<br>" + str(i) + "  " + c1.firstChild.data
    except:
        print("exce")
