import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import random

# lenstronomy module import
import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse
gauss = GaussianEllipse()



# Define a specific cosmology
#from astropy.cosmology import FlatLambdaCDM
#cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)


# Instrument setting from pre-defined configurations
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid

Euclid_g = Euclid(band='VIS', psf_type='GAUSSIAN', coadd_years=6)
kwargs_g_band = Euclid_g.kwargs_single_band()
Euclid_r = Euclid(band='VIS', psf_type='GAUSSIAN', coadd_years=6)
kwargs_r_band = Euclid_r.kwargs_single_band()
Euclid_i = Euclid(band='VIS', psf_type='GAUSSIAN', coadd_years=6)
kwargs_i_band = Euclid_i.kwargs_single_band()

from pyHalo.preset_models import ULDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.single_realization import SingleHalo

from lenstronomy.LensModel.Profiles.cnfw import CNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.uldm import Uldm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

def simulate(image_galaxy,sigma_v,source_pos_xx,source_pos_yy,source_ang,zlens=0.5,zsource=1.0):


    # pyHalo inputs to generate single ULDM halos
    axion_mass = -21
    profile_arg = {'log10_m_uldm': axion_mass,
                     'uldm_plaw': 1/3,
                     'scale_nfw':False,
                     'evaluate_mc_at_zlens':True,
                     'c_scatter':False}

    # define the lens model of the main deflector
    main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
    kwargs_lens_main = {'theta_E': 1.5, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0.}
    kwargs_shear = {'gamma1': 0.05, 'gamma2': 0}
    lens_model_macro = [main_halo_type, 'SHEAR']
    kwargs_lens = [kwargs_lens_main, kwargs_shear]


    log10_m_uldms, uldm_plaw, scale_nfw = axion_mass, 1/3, False

    realizationsULDM = ULDM(zlens, zsource, 
                           log10_m_uldm=log10_m_uldms, uldm_plaw=uldm_plaw, scale_nfw=scale_nfw)

    print(r'ULDM realization with log(m)=%s contains %s halos' 
              % (log10_m_uldms, len(realizationsULDM.halos)))


    lens_model_list, lens_redshift_array, kwargs_halos, numerical_deflection_class = realizationsULDM.lensing_quantities()
    astropy_instance = realizationsULDM.astropy_instance

    # cored halo + shear macromodel
    kwargs_macromodel = kwargs_lens

    lens_model_list_macro = lens_model_macro #[main_halo_type, 'SHEAR']#['CNFW','ULDM', 'SHEAR']

    lens_model_list_full = lens_model_list_macro + lens_model_list
    lens_redshift_list_full = [zlens, zlens] + list(lens_redshift_array)
    kwargs_lens_full = kwargs_macromodel + kwargs_halos
    kwargs_lens_list = kwargs_lens_full

    kwargs_lens = kwargs_lens_full

###############################################################################

    kwargs_model_physical = {'lens_model_list': lens_model_list_full,  # list of lens models to be used
                              'lens_redshift_list': lens_redshift_list_full,  # list of redshift of the deflections
                              'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'],  # list of unlensed light models to be used
                              'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used
                              'source_redshift_list': [1.0],  # list of redshfits of the sources in same order as source_light_model_list
                              'cosmo': astropy_instance,  # astropy.cosmology instance
                              'z_source_convention': 2.5,  # source redshfit to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
                              'z_source': 2.5,}  # redshift of the default source (if not further specified by 'source_redshift_list') and also serves as the redshift of lensed point sources

    #kwargs_mass = [{'sigma_v': sigma_v, 'center_x': 0, 'center_y': 0, 'e1': 0.0, 'e2': 0}]

    kwargs_model_postit = kwargs_model_physical

    #######################################################################
    numpix = 64  # number of pixels per axis of the image to be modelled

    # here we define the numerical options used in the ImSim module. 
    # Have a look at the ImageNumerics class for detailed descriptions.
    # If not further specified, the default settings are used.
    kwargs_numerics = {'point_source_supersampling_factor': 1}


    #######################################################################
    sim_g = SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=kwargs_model_postit)
    sim_r = SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model_postit)
    sim_i = SimAPI(numpix=numpix, kwargs_single_band=kwargs_i_band, kwargs_model=kwargs_model_postit)

    # return the ImSim instance. With this class instance, you can compute all the
    # modelling accessible of the core modules. See class documentation and other notebooks.
    imSim_g = sim_g.image_model_class(kwargs_numerics)
    imSim_r = sim_r.image_model_class(kwargs_numerics)
    imSim_i = sim_i.image_model_class(kwargs_numerics)

    source_scale = 0.0025

    X,Y = source_pos_xx, source_pos_yy

    # g-band
    # lens light
    image_data_g = image_galaxy[:,:,0].astype(float)
    median_g = np.median(image_galaxy[:,:,0][:50, :50].astype(float))
    image_data_g -= median_g

    #kwargs_lens_light_mag_g = [{'magnitude': 16, 'image': image_gauss, 'scale': lens_scale, 'phi_G': 0, 'center_x': 0., 'center_y': 0}]
    # lens light
    kwargs_lens_light_mag_g = [{'magnitude': 17, 'R_sersic': 0.4, 'n_sersic': 2.3, 'e1': 0, 'e2': 0.05, 'center_x': 0, 'center_y': 0},
    {'magnitude': 28, 'R_sersic': 1.5, 'n_sersic': 1.2, 'e1': 0, 'e2': 0.3, 'center_x': 0, 'center_y': 0}]
    kwargs_source_mag_g = [{'magnitude': 22, 'image': image_data_g, 'scale': source_scale, 'phi_G': source_ang, 'center_x': X, 'center_y': Y}]

    # and now we define the colors of the other two bands

    # r-band
    g_r_source = 1  # color mag_g - mag_r for source
    g_r_lens = -1  # color mag_g - mag_r for lens light
    #g_r_ps = 0
    kwargs_lens_light_mag_r = copy.deepcopy(kwargs_lens_light_mag_g)
    kwargs_lens_light_mag_r[0]['magnitude'] -= g_r_lens

    image_data_r = image_galaxy[:,:,1].astype(float)
    median_r = np.median(image_galaxy[:,:,1][:50, :50].astype(float))
    image_data_r -= median_r

    kwargs_source_mag_r = [{'magnitude': 20 - g_r_source, 'image': image_data_r, 'scale': source_scale, 'phi_G': source_ang, 'center_x': X, 'center_y': Y}]

    # i-band
    g_i_source = 2
    g_i_lens = -2
    #g_i_ps = 0
    kwargs_lens_light_mag_i = copy.deepcopy(kwargs_lens_light_mag_g)
    kwargs_lens_light_mag_i[0]['magnitude'] -= g_i_lens

    image_data_i = image_galaxy[:,:,2].astype(float)
    median_i = np.median(image_galaxy[:,:,2][:50, :50].astype(float))
    image_data_i -= median_i

    kwargs_source_mag_i = [{'magnitude': 20 - g_i_source, 'image': image_data_i, 'scale': source_scale, 'phi_G': source_ang, 'center_x': X, 'center_y': Y}]

    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g,_ = sim_g.magnitude2amplitude(kwargs_lens_light_mag_g, kwargs_source_mag_g)
    kwargs_lens_light_r, kwargs_source_r,_ = sim_r.magnitude2amplitude(kwargs_lens_light_mag_r, kwargs_source_mag_r)
    kwargs_lens_light_i, kwargs_source_i,_ = sim_i.magnitude2amplitude(kwargs_lens_light_mag_i, kwargs_source_mag_i)

    #added then removed by me
    #kwargs_lens = sim_g.physical2lensing_conversion(kwargs_mass=kwargs_mass)

    image_g = imSim_g.image(kwargs_lens, kwargs_source_g, kwargs_lens_light_g)
    image_r = imSim_r.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r)
    image_i = imSim_i.image(kwargs_lens, kwargs_source_i, kwargs_lens_light_i)

    # add noise
    image_g += sim_g.noise_for_model(model=image_g)
    image_r += sim_r.noise_for_model(model=image_r)
    image_i += sim_i.noise_for_model(model=image_i)

    # save to output
    img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
    img[:,:,0] = image_g#plot_util.sqrt(image_g, scale_min=0, scale_max=100)
    img[:,:,1] = image_r #plot_util.sqrt(image_r, scale_min=0, scale_max=100)
    img[:,:,2] = image_i#plot_util.sqrt(image_i, scale_min=0, scale_max=100)

    return img

################################################
######### Load real galaxies
################################################
# To get the images and labels from file
with h5py.File('../data/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    typ = np.array(F['ans'])
    z = np.array(F['redshift'])

unbarred_spiral = np.where(typ == 6)
images_ref = images[unbarred_spiral]
z_ref = z[unbarred_spiral]

indx_img_zp1 = np.where(z_ref < 0.02)
img_zp1 = images_ref[indx_img_zp1]
# good galaxies, checked by eye
arr = [2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,54,55]
#################################################

number_of_sims = int(2.5e4)

for i in range(number_of_sims):
    index = np.random.randint(0,len(arr))

    sigma_v,source_pos_xx,source_pos_yy, source_ang = np.random.normal(260,20), np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3), np.random.uniform(-np.pi,np.pi)

    sim = simulate(img_zp1[arr[index]],sigma_v,source_pos_xx,source_pos_yy,source_ang)

    if i % 10 == 0:
        np.save('val/axion/sim_'+ str(random.getrandbits(128)),(np.array(sim).clip(min=0)/np.max(sim)),allow_pickle=True)
    else:
        np.save('train/axion/sim_'+ str(random.getrandbits(128)),(np.array(sim).clip(min=0)/np.max(sim)),allow_pickle=True)

