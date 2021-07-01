import sqlite3
import pandas as pd
import dirfuncs
conn = sqlite3.connect('data/hcs_database.db')
#conn = sqlite3.connect('hcs_database.db')

##TODO this should be moved to hcs_database.py
base_dir = dirfuncs.guess_data_dir()
concessions_csv = base_dir + 'concession_inventory.csv'
concession_df = pd.read_csv(concessions_csv)
assessment_year_dict = dict(zip(concession_df.app_key, concession_df.use_year))


leaf_nodes_dict={
'Bumitama_PTDamaiAgroSejahtera':10,
'Bumitama_PTHungarindoPersada':6,
'PTMitraNusaSarana':10,
'makmur_abadi':10,
'sawit_perdana':6,
'aneka_sawit':10,
'PTMentariPratama':6,
'PTSukajadiSawitMekar':10,
'PTLabontaraEkaKarsa':10,
'adi_perkasa': 8

}
features_dict={
'Bumitama_PTDamaiAgroSejahtera':0.33,
'Bumitama_PTHungarindoPersada':0.33,
'PTMitraNusaSarana':0.65,
'makmur_abadi':0.33,
'sawit_perdana':0.8,
'aneka_sawit':0.65,
'PTMentariPratama':0.65,
'PTSukajadiSawitMekar':0.8,
'PTLabontaraEkaKarsa':0.8,
'adi_perkasa': 0.65
}
estimators_dict={
'Bumitama_PTDamaiAgroSejahtera':500,
'Bumitama_PTHungarindoPersada':500,
'PTMitraNusaSarana':625,
'makmur_abadi':500,
'sawit_perdana':500,
'aneka_sawit':750,
'PTMentariPratama':500,
'PTSukajadiSawitMekar':625,
'PTLabontaraEkaKarsa':500,
'adi_perkasa': 625
}

data_context_dict =  {
##CONCESSIONS
'app_muba':'Sumatra',
'app_riau': 'Sumatra',
'app_oki' : 'Sumatra',
'app_jambi' : 'Sumatra',
'crgl_stal' : 'Sumatra',
'gar_pgm':'Kalimantan',
'app_kalbar':'Kalimantan',
'app_kaltim':'Kalimantan',
'Bumitama_PTDamaiAgroSejahtera':'Kalimantan',
'Bumitama_PTGemilangMakmurSubur':'Kalimantan' ,
'Bumitama_PTHungarindoPersada':'Kalimantan',
'PTMitraNusaSarana':'Kalimantan',
'adi_perkasa':'Papua',
'agro_lestari':'Papua',
'makmur_abadi':'Kalimantan',
'PTLestariAbadiPerkasa':'Kalimantan',
'sawit_perdana':'Kalimantan',
'PTGlobalindoAlamPerkasa':'Kalimantan',
'aneka_sawit':'Kalimantan',
'PTMentariPratama':'Kalimantan',
'PTSukajadiSawitMekar':'Kalimantan',
'PTLabontaraEkaKarsa':'Kalimantan',
'multipersada_gatramegah':'Kalimantan',
'musim_mas':'Sumatra',
'unggul_lestari':'Kalimantan',
'mukti_prakarsa':'Kalimantan',
'agro_mandiri':'Kalimantan',
'PTAgroAndalan':'Kalimantan',
'betun_kerihun':'Kalimantan',
'gunung_palung':'Kalimantan',
'tekukur_indah':'Kalimantan',
'varia_mitra_andalan':'Papua',
'agro_lestari':'Papua',
'tunas_sawwaerma':'Papua',

##ISLANDS
'Kalimantan':'Kalimantan',
'Papua':'Papua',
'Sumatra':'Sumatra',

##PROVINCES
'West_Kalimantan': 'Kalimantan',
'East_Kalimantan': 'Kalimantan',
'South_Kalimantan': 'Kalimantan',
'Central_Kalimantan': 'Kalimantan',

'Jambi': 'Sumatra',
'Lampung':'Sumatra',
'Riau':'Sumatra',
'West_Sumatra': 'Sumatra',
'South_Sumatra': 'Sumatra',
'Bengkulu': 'Sumatra',
'North_Sumatra': 'Sumatra',
'Aceh': 'Sumatra',

##supplemental classes
'impervious':'supplementary_class',
'oil_palm':'supplementary_class',
'agriculture':'supplementary_class',
'forest':'supplementary_class',
'coconut':'supplementary_class',
'pulp_and_paper':'supplementary_class',
'water':'supplementary_class',
          }


model_performance_columns = ['concession' , 'bands' , 'score_type'  , 'class_scheme' , 'score' , 'score_weighted' ,
                                   'two_class_score' , 'two_class_score_weighted' , 'training_concessions' , 'max_depth',
                                   'max_leaf_nodes' , 'max_features' , 'n_estimators' , 'training_sample_rate' , 'resolution' , 'kappa', 'kappa_3']

shapefile_base = dirfuncs.guess_data_dir() + 'stratified_shapefiles/'
province_shapfile_dir = dirfuncs.guess_data_dir() + 'province_shapefiles/'

shapefiles = {'app_kalbar': shapefile_base + 'Kalbar_DTK_Stratification.shp',
               'app_jambi': shapefile_base + 'Jambi_WKS_Stratification.shp',
               'app_kaltim': shapefile_base + 'Kaltim_KHL_Stratification.shp',
               'app_muba': shapefile_base + 'Muba_BPP2_Stratification.shp',
               'app_riau': shapefile_base + 'Riau_MSK_SK_Stratification.shp',
               'app_oki': shapefile_base + 'OKI_BMH_Stratification.shp',
                'crgl_stal': shapefile_base + 'crgl_stal.shp',

                'West_Kalimantan': province_shapfile_dir + 'Kalimantan_Barat.shp',
              'East_Kalimantan': province_shapfile_dir + 'Kalimantan_Timur.shp',
              'South_Kalimantan': province_shapfile_dir + 'Kalimantan_Selatan.shp',
              'Central_Kalimantan': province_shapfile_dir + 'Kalimantan_Tengah.shp',

                'Jambi': province_shapfile_dir + 'Jambi_Province.shp',
                'Lampung': province_shapfile_dir + 'Lampung.shp',
                'Riau': province_shapfile_dir + 'Riau_clip.shp',
                'West_Sumatra': province_shapfile_dir + 'Sumatera_Barat.shp',
                'South_Sumatra': province_shapfile_dir + 'Sumatera_Selatan_clip.shp',
                'Bengkulu': province_shapfile_dir + 'Bengkulu.shp',
                'North_Sumatra': province_shapfile_dir + 'Sumatera_Utara.shp',
                'Aceh': province_shapfile_dir + 'Aceh.shp',

                'Papua': province_shapfile_dir + 'Papua.shp',
                'betun_kerihun': shapefile_base + 'betun_kerihun.shp',
                'gunung_palung': shapefile_base + 'gunung_palung.shp',
               'app_all': shapefile_base + 'app_all'}

#This method should only be used once or when you want to wipe out any existing data
def init_database():
   c = conn.cursor()


   ## Create table
   # c.execute('DROP TABLE model_performance_log')
   # c.execute('''CREATE TABLE model_performance_log
   #              (concession text, bands text, score_type text , class_scheme text, score real, score_weighted real,
   #                                two_class_score real, two_class_score_weighted real, training_concessions text, max_depth int,
   #                                max_leaf_nodes int, max_features real, n_estimators int, training_sample_rate real)''')

   addColumn = "ALTER TABLE model_performance_log ADD COLUMN resolution int"
   #addColumn = "ALTER TABLE model_performance_log ADD COLUMN score_weighted real"
   c.execute(addColumn)
   addColumn = "ALTER TABLE model_performance_log ADD COLUMN kappa real"
   c.execute(addColumn)
   # addColumn = "ALTER TABLE model_performance_log ADD COLUMN two_class_score_weighted real"
   # c.execute(addColumn)
   # addColumn = "ALTER TABLE model_performance_log ADD COLUMN training_concessions text"
   # c.execute(addColumn)
   # addColumn = "ALTER TABLE model_performance_log ADD COLUMN max_depth int"
   # c.execute(addColumn)
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
    c.execute('DELETE FROM model_performance_log ' )#where length(training_concessions) = 116 ')
    conn.commit()

def get_all_model_performance():
    df = pd.read_sql_query("SELECT * FROM model_performance_log", conn)
    #df = pd.read_sql_query("SELECT max(length(training_concessions)) FROM model_performance_log", conn)
    print('ROWS:  ', len(df))
    return df

def get_max_model_run(concession, bands):
    c = conn.cursor()
    c.execute("SELECT * FROM model_performance_log where two_class_score_weighted = ( SELECT max(two_class_score_weighted) from model_performance_log where max_leaf_nodes < 13 and max_features <.81 and class_scheme='3CLASS' and concession = ? and bands = ?)" ,  (concession, bands) )
    rows = c.fetchall()
    print('ROWS:  ', len(rows))
    for row in rows:
        row_dict = dict(zip(model_performance_columns,row))
    return row_dict

def get_max_model_run(concession):
    c = conn.cursor()
    #c.execute("SELECT * FROM model_performance_log where two_class_score_weighted = ( SELECT max(two_class_score_weighted) from model_performance_log where max_leaf_nodes < 13 and max_features <.81 and class_scheme='3CLASS' and concession = ?)" ,  (concession) )
    c.execute(
        "SELECT * FROM model_performance_log where max_leaf_nodes < 13 and max_features <.81 and class_scheme='3CLASS' and concession = ?  order by round(kappa_3,1) desc, n_estimators desc, two_class_score_weighted desc ",
        (concession))

    rows = c.fetchone()
    row_dict = dict(zip(model_performance_columns,list(rows)))

    return row_dict

def get_best_training_sample_rate(concession):
    return get_max_model_run(concession)['training_sample_rate']


def get_best_max_features(concession):
    return features_dict[concession]
    #return get_max_model_run(concession)['max_features']

def get_best_max_leaf_nodes(concession):
    return leaf_nodes_dict[concession]
    #return get_max_model_run(concession)['max_leaf_nodes']

def get_best_number_estimators(concession):
    return estimators_dict[concession]
    #return get_max_model_run(concession)['n_estimators']

def get_best_max_depth(concession):
    return get_max_model_run(concession)['max_depth']

def get_best_bands(concession):
    x= get_max_model_run(concession)['bands']
    bands = x.replace('[', '').replace(']','').replace('\'','')
    return bands.split(', ')

def get_concession_assessment_year(concession):
    return assessment_year_dict[concession]

def get_best_training_sample_rate_byBand(concession, bands):
    return get_max_model_run(concession, bands)['training_sample_rate']


def get_best_max_features_byBand(concession, bands):
    return get_max_model_run(concession, bands)['max_features']

def get_best_max_leaf_nodes_byBand(concession, bands):
    return get_max_model_run(concession, bands)['max_leaf_nodes']

def get_best_number_estimators_byBand(concession, bands):
    return get_max_model_run(concession, bands)['n_estimators']

def get_best_max_depth_byBand(concession, bands):
    return get_max_model_run(concession, bands)['max_depth']

def get_best_metric(concession):
    return get_max_model_run(concession)['score_type']

def get_best_scheme(concession):
    return get_max_model_run(concession)['class_scheme']

if __name__ == "__main__":
    print('in main')
    #init_database()
    print(get_all_model_performance())
   #  conn = sqlite3.connect('hcs_database.db')
   #  base_dir = dirfuncs.guess_data_dir()
   #  resultfile = base_dir + 'result.06022020_server.csv'
   #  df = pd.read_csv(resultfile)
   #  df.to_sql('model_performance_log', conn, if_exists='append', index=False)
   # # print(get_all_model_performance())
   #  get_all_model_performance().to_csv(resultfile, index=False)
   # print(get_best_bands(['Bumitama_PTHungarindoPersada']))

    init_database()
   # delete_model_performance()
    print(get_all_model_performance())
    conn.close()
    #print(all_bands)