import sqlite3
import pandas as pd
import dirfuncs
#conn = sqlite3.connect('data/hcs_database.db')

maps_dict = {'app_jambi': 'ft:1bgkWL4VgYSgfAupZmVGcnXJJqMmvyBtl3_VgfyVV',
             'app_kalbar': 'ft:16yV7XDfeb1fGhH-N68CIetxd0FW8OvsTHdB7I4ka',
             'app_muba': 'ft:12FXxeD_Thxl84MQ-Bk1E0AEhL8N3ORg45uCTxXIx',
             'app_oki': 'ft:1GmCNR3duB82VE9DbUZkMZ_aVmVciYvrOVkJLFLsL',
             'app_kaltim': 'ft:18UtezGIZKNZywbxIt18pAXObwI1ok60Amc-XPvx1',
             'app_riau': 'ft:1dA4oDlvcY0KipkR7XI4CNwFyJc0uh807WRFA4ju5',
             'wilm_cal': 'ft:1eHm5seFIE_CydzRgNenfAj20Qkh_SlqQkEtbxcLH',
             'gar_pgm': 'ft:1YG1x2BRB63C4CRY6oBUwUxzbo569IN37nW9RMcA0',
             'crgl_stal': 'ft:12lUjvcMXXuSrSS4w1eMVCgjvTjksKY7FS453eHda',
             'app_all': 'ft:11bZ-vOrh6Qf5Zcf7D88eGIpIUVe7WGguXKD8YEDa'}

plots_dict = {'crgl_stal': '',
              'app': '',
              'wilm_cal': '',
              'gar_pgm': ''}


study_areas = {'app_muba': {"type":"Polygon","coordinates":[[[102.9583740234375,-2.269723057075878],
                                                             [102.9693603515625,-2.533163135991036],
                                                             [103.4857177734375,-2.533163135991036],
                                                             [103.46923828125,-2.280700724265905]]],"evenOdd":'true'},
               'app_kalbar': {"type":"Polygon","coordinates":[[[110.1214599609375,-0.7776688790136811],
                                                               [109.8193359375,-0.5455983301132842],
                                                               [109.55291748046875,-0.6582019616289609],
                                                               [109.86465454101562,-0.9136098500495502]]],"evenOdd": 'true'},
               'app_oki': {"type":"Polygon","coordinates":[[[104.8699951171875,-2.8569008341164004],
                                                            [105.347900390625,-4.183708325021417],
                                                            [106.14990234375,-3.8439680300422023],
                                                            [105.6005859375,-2.6648633477255825]]],"evenOdd":'true'},
               'app_jambi': {"type":"Polygon","coordinates":[[[102.359619140625,-0.68277635408023],
                                                              [102.5518798828125,-1.9019255518982194],
                                                              [104.468994140625,-1.5340496285948395],
                                                              [104.095458984375,-0.5564409938843804]]],"evenOdd":'true'},
               'app_kaltim': {"type":"Polygon","coordinates":[[[115.88653564453125,-0.7368837547081416],
                                                               [115.88653564453125,-0.9510937861260604],
                                                               [116.0101318359375,-0.9401089541971663],
                                                               [116.004638671875,-0.7313910415641723]]],"evenOdd":'true'},
               'app_riau': {"type":"Polygon","coordinates":[[[102.64526400715113,-0.6185033744297275],
                                                             [103.46923861652613,-0.036239994136064956],
                                                             [101.85424838215113,1.6773875302689043],
                                                             [101.21704135090113,1.1611902706982198]]],"evenOdd":'true'}}

gee_dir = 'users/rheilmayr/indonesia/'
shapefile_base = dirfuncs.guess_data_dir() + 'stratified_shapefiles/'
app_rasters = {'app_kalbar': gee_dir + 'Kalbar_DTK_Stratification',
               'app_jambi': gee_dir + 'Jambi_WKS_Stratification',
               'app_kaltim': gee_dir + 'Kaltim_KHL_Stratification',
               'app_muba': gee_dir + 'Muba_BPP2_Stratification',
               'app_riau': gee_dir + 'Riau_MSK_SK_Stratification',
               'app_oki': gee_dir + 'OKI_BMH_Stratification',
               'app_all': gee_dir + 'app_all'}

shapefiles = {'app_kalbar': shapefile_base + 'Kalbar_DTK_Stratification.shp',
               'app_jambi': shapefile_base + 'Jambi_WKS_Stratification.shp',
               'app_kaltim': shapefile_base + 'Kaltim_KHL_Stratification.shp',
               'app_muba': shapefile_base + 'Muba_BPP2_Stratification.shp',
               'app_riau': shapefile_base + 'Riau_MSK_SK_Stratification.shp',
               'app_oki': shapefile_base + 'OKI_BMH_Stratification.shp',
               'app_all': shapefile_base + 'app_all'}

rasters = {'app_kalbar': gee_dir + 'Kalbar_DTK_Stratification',
           'app_jambi': gee_dir + 'Jambi_WKS_Stratification',
           'app_kaltim': gee_dir + 'Kaltim_KHL_Stratification',
           'app_muba': gee_dir + 'Muba_BPP2_Stratification',
           'app_riau': gee_dir + 'Riau_MSK_SK_Stratification',
           'app_oki': gee_dir + 'OKI_BMH_Stratification',
           'app_all': gee_dir + 'app_all',
           'crgl_stal': gee_dir + 'crgl_stal',
           'gar_pgm': gee_dir + 'gar_pgm',
           'nbpol_ob': gee_dir + 'nbpol_ob',
           'wlmr_calaro': gee_dir + 'wlmr_calaro'}

def init_database():
    c = conn.cursor()

    addColumn = "ALTER TABLE model_performance_log ADD COLUMN resolution int"

    c.execute(addColumn)
    update = "UPDATE model_performance_log set resolution=60"
    c.execute(update)
    # Create table
    # c.execute('DROP TABLE model_performance_log')
    # c.execute('''CREATE TABLE model_performance_log
    #              (concession text, bands text, score_type text , class_scheme text, score real, score_weighted real,
    #                                two_class_score real, two_class_score_weighted real, training_concessions text, max_depth int,
    #                                max_leaf_nodes int, max_features real, n_estimators int, training_sample_rate real)''')

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.


def save_model_performance(rows):
    c = conn.cursor()
    rows.to_sql(name='model_performance_log', con=conn, if_exists='append', index=False)
    #c.executemany('INSERT INTO model_performance_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    conn.commit()

def delete_model_performance():
    c = conn.cursor()
    c.execute('DELETE FROM model_performance_log where resolution=30 ')
    conn.commit()

def get_all_model_performance():
    df = pd.read_sql_query("SELECT * FROM model_performance_log", conn)
    return df

if __name__ == "__main__":
    print('in main')
    #init_database()
    #delete_model_performance()
#    print(get_all_model_performance().tail())
  #  conn.close()
    print(shapefiles['app_riau'])