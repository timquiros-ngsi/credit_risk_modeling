labels_dict = {
   "loantype":{
      "AL":0,
      "AP":1,
      "AS":2,
      "BL":3,
      "CL":4,
      "CP":5,
      "EM":6,
      "LL":7,
      "MT":8,
      "NL":9,
      "OP":10,
      "PC":11,
      "PL":12,
      "SL":13
   },
   "me_svc_stat":{
      "AC":0,
      "CO":1,
      "DF":2,
      "DI":3,
      "NO":4,
      "OP":5,
      "RE":6,
      "RS":7
   },
   "pnpbillmode":{
      "NONE":0,
      "PBM00":1,
      "PBM01":2,
      "PBM02":3,
      "PBM03":4,
      "PBM04":5,
      "PBM05":6,
      "PBM06":7,
      "PBM07":8
   }
}

features = ['Term', 'Loan type', 'me_age', 'loanamt', 'me_svc_stat',
       'grosspay1', 'pnpbillmode', 'me_app_place', 'me_mem_pay_type',
       'unit_code', 'place_cd', 'me_rank', 'me_member_stat', 'me_cap_con',
       'branch_code', 'me_sex', 'me_brn_srv', 'me_lack_req',
       'me_svc_class', 'loanappl', 'outsloanamt', 'paymode',
       'me_member_type', 'me_civil', 'me_hcity_town', 'nlp2',
       'lneffintrate', 'lnintratepa']