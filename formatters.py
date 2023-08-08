def multiples_tbl_formatter(multiples_table_obj):
    rows = [
        "   ".join(["              ", *[f"{year:^6}" for year in multiples_table_obj["EBIT/Revenue"].keys()]]),
        "   ".join(["EBIT/Revenue  ", *[f"{val:06.3f}" for val in multiples_table_obj["EBIT/Revenue"].values()]]),
        "   ".join(["EBITDA/Revenue", *[f"{val:06.3f}" for val in multiples_table_obj["EBITDA/Revenue"].values()]]),
        "   ".join(["Leverage Ratio", *[f"{val:06.3f}" for val in multiples_table_obj["Leverage Ratio"].values()]])
    ]
    table = "\n".join(rows)
    return table


def risk_tbl_formatter(risk_table_obj):
    components = ["structural", "growth", "sensitivity", "overall"]
    rows = [
        "  ".join([f"{h:<16}" for h in ("Risk component", "Level", "Weight", "Score")]),
        *[
            "  ".join([
                f'{c.title() + " Risk":<16}',
                f'{risk_table_obj[c].get("level", ""):<16}',
                f'{risk_table_obj[c].get("weight", ""):<16}',
                f'{risk_table_obj[c].get("score", ""):<16}',
            ]) for c in components
        ]
    ]
    table = "\n".join(rows)
    return table


def barriers_to_entry_formatter(barriers_to_entry_obj):
    return "\n".join([f"{k + ':':<22} {v.title()}" for k, v in barriers_to_entry_obj.items()])


# test_multiples_table_obj = {
#     "latestPubYear": "2021",
#     "EBIT/Revenue": {
#         "2007": 10.255181230975822,
#         "2008": 11.232272598187462,
#         "2009": 12.908739618557536,
#         "2010": 13.43318791152652,
#         "2011": 14.897759141396612,
#         "2012": 20.130363739480163,
#         "2013": 20.024473899498854,
#         "2014": 20.045942031838432,
#         "2015": 23.127274130081883,
#         "2016": 23.57577153124632,
#         "2017": 23.62128780176576,
#         "2018": 24.982562821040126,
#         "2019": 25.048396687027648,
#         "2020": 24.656230087022823,
#         "2021": 21.499992127359626,
#         "3-Year": 23.7348729671367,
#         "5-Year": 23.961693904843198,
#         "10-Year": 22.671229485636164
#     },
#     "EBITDA/Revenue": {
#         "2007": 15.1282825876773,
#         "2008": 15.618977942369872,
#         "2009": 16.91601223808106,
#         "2010": 17.722778827709337,
#         "2011": 19.03084606687896,
#         "2012": 23.983988921398513,
#         "2013": 23.771189414276098,
#         "2014": 23.625858042353755,
#         "2015": 26.824720651583867,
#         "2016": 27.43775378595352,
#         "2017": 27.220461713484383,
#         "2018": 28.310207655958695,
#         "2019": 28.3949686148263,
#         "2020": 41.11663987390589,
#         "2021": 36.80257239733089,
#         "3-Year": 35.43806029535436,
#         "5-Year": 32.36897005110123,
#         "10-Year": 28.74883610710719
#     },
#     "Leverage Ratio": {
#         "2007": 4.432760164798806,
#         "2008": 3.997989422495005,
#         "2009": 3.5191550545743215,
#         "2010": 3.7569432882541225,
#         "2011": 3.1350568759627304,
#         "2012": 2.4531859332001194,
#         "2013": 2.650101376035598,
#         "2014": 2.644299969582721,
#         "2015": 2.249660623025392,
#         "2016": 2.093554583053665,
#         "2017": 2.6013561697902605,
#         "2018": 2.423443623258813,
#         "2019": 2.62036750598746,
#         "2020": 1.831380588887279,
#         "2021": 2.1434431107068908,
#         "3-Year": 2.1983970685272096,
#         "5-Year": 2.3239981997261405,
#         "10-Year": 2.37107934835282
#     },
#     "parentInfo": None
# }
#
# tbl = multiples_tbl_formatter(test_multiples_table_obj)
# print(tbl)
#
# test_risk_table = {
#     "overall": {
#         "score": 4.99,
#         "level": "Medium"
#     },
#     "growth": {
#         "score": 6.16,
#         "level": "High",
#         "weight": 25
#     },
#     "sensitivity": {
#         "score": 3.86,
#         "level": "Low",
#         "weight": 50
#     },
#     "structural": {
#         "score": 6.09,
#         "level": "High",
#         "weight": 25
#     }
# }
#
# tbl = risk_tbl_formatter(test_risk_table)
# print(tbl)


# test_barriers_to_entry = {
#     "Competition": "high",
#     "Concentration": "low",
#     "Life Cycle Stage": "mature",
#     "Capital Intensity": "low",
#     "Technology Change": "medium",
#     "Regulation & Policy": "medium",
#     "Industry Assistance": "low"
# }
#
# result = barriers_to_entry_formatter(test_barriers_to_entry)
# print(result)
