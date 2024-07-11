#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:54:39 2021

@author: arefe
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
#import pandas as pd
import healpy as hp
import time
import warnings
#from pixell import reproject
#from classy import Class
import spherical
import quaternionic
from scipy.integrate import trapz



def beam2bl(beam, theta, lmax):
    """Computes a transfer (or window) function b(l) in spherical
    harmonic space from its circular beam profile b(theta) in real
    space.
    Parameters
    ----------
    beam : array
        Circular beam profile b(theta).
    theta : array
        Radius at which the beam profile is given. Has to be given
        in radians with same size as beam.
    lmax : integer
        Maximum multipole moment at which to compute b(l).
    Returns
    -------
    bl : array
        Beam window function b(l).
    """

    nx = len(theta)
    nb = len(beam)
    if nb != nx:
        raise ValueError("Beam and theta must have same size!")

    x = np.cos(theta)
    st = np.sin(theta)
    window = np.zeros(lmax + 1, dtype = np.complex128)

    p0 = np.ones(nx)
    p1 = np.copy(x)

    window[0] = trapz(beam * p0 * st, theta)
    window[1] = trapz(beam * p1 * st, theta)

    for l in np.arange(2, lmax + 1):
        p2 = x * p1 * (2 * l - 1) / l - p0 * (l - 1) / l
        window[l] = trapz(beam * p2 * st, theta)
        p0 = p1
        p1 = p2

    window *= 2 * np.pi
    return window


def integrate (beam, theta):
    nx = len(theta)
    nb = len(beam)
    if nb != nx:
        raise ValueError("Beam and theta must have same size!")

    #x = np.cos(theta)
    st = np.sin(theta)
    #window = np.zeros(lmax + 1, dtype = np.complex128)

    #p0 = np.ones(nx)
    #p1 = np.copy(x)

    ans = 2 * np.pi * trapz(beam * st, theta)
    
    
    
    return ans 

def integrate_dir (beam, theta, phi):
    nx = len(theta)
    nb = len(beam)
    ny = len(phi)
    if nb != nx:
        raise ValueError("Beam and theta must have same size!")

    #x = np.cos(theta)
    st = np.sin(theta)
    #window = np.zeros(lmax + 1, dtype = np.complex128)

    p0 = np.ones(nx)
    #p1 = np.copy(x)

    ans_theta = trapz(beam * st, theta)
    ans_phi = trapz(p0, phi )
    ans = ans_theta * ans_phi
    
    return ans 



def gabor(freq,sigma,theta):
    
    arg=-(theta**2)/(2*sigma*sigma) + 1.j * freq*theta
    
    g=np.exp(arg)
    g/=2*np.pi*sigma*sigma
    return g


def morlet (f, sigma, theta, lmax):
    wv=gabor(f,sigma,theta)
    wvm=gabor(0,sigma, theta)
    B=integrate(wv,theta)/integrate(wvm,theta)
    mor=wv-B*wvm
    #norm = np.mean(np.abs(mor)**2*np.sin(theta))
    #mor = mor/norm
    
    return mor


 

def morlet_arr (resol, jmax, lmax, theta_bin) :
    mor=[]
    fl2beam=[]
    
    theta=np.linspace(0,np.pi,theta_bin)
    for j in range(jmax):
        sigma1 = (0.8*resol*2**j)
        freq1=(3.0*np.pi) /(4.0*resol*2**j)
        morf = morlet (freq1, sigma1, theta, lmax)
        #print ("ok")
        #norm = np.sum((np.abs (morf))**2 * np.sin(theta)*np.pi/theta_bin)
        #morf /= norm
        #print ("okk")
        mor_l = beam2bl(morf, theta, lmax)
        #norm = np.sqrt(np.sum((np.abs (morf))**2 * np.sin(theta)*np.pi/theta_bin))
        #print ("hello")
        mor.append(mor_l)
    
    return mor



def gabor_prj(freq,sigma,theta):
    
    arg=-(4*np.tan(theta/2)**2)/(2*sigma*sigma) + 2.j * freq*np.tan(theta/2)
    
    g=np.exp(arg)
    g/=2*np.pi*sigma*sigma
    return g


def morlet_prj (f, sigma, theta, lmax):
    wv=gabor_prj(f,sigma,theta)
    wvm=gabor_prj(0,sigma, theta)
    B=integrate(wv/(1+np.cos(theta)),theta)/integrate(wvm/(1+np.cos(theta)),theta)
    mor=wv-B*wvm
    #norm = np.mean(np.abs(mor)**2*np.sin(theta))
    #mor = mor/norm
    print ("hihi")
    return mor


 

def morlet_arr_prj (resol, jmax, lmax, theta_bin) :
    morl=[]
    fl2beam=[]
    
    theta=np.linspace(0,np.pi-0.0001,theta_bin)
    for j in range(jmax):
        sigma1 = (0.8*resol*2**j)
        freq1=(3.0*np.pi) /(4.0*resol*2**j)
        morf = morlet_prj (freq1, sigma1, theta, lmax)
        #print ("ok")
        #norm = np.sum((np.abs (morf))**2 * np.sin(theta)*np.pi/theta_bin)
        #morf /= norm
        #print ("okk")
        mor_l = beam2bl(morf, theta, lmax)
        #norm = np.sqrt(np.sum((np.abs (morf))**2 * np.sin(theta)*np.pi/theta_bin))
        #print ("hello")
        morl.append(mor_l)
    
    return morl



def convolve_dir (field_l, wavelet_l, lmax, theta_arr , phi_arr):
    wigner = spherical.Wigner(lmax)
    R = quaternionic.array.from_spherical_coordinates(theta_arr, phi_arr)
    D = wigner.D(R)
    h = np.zeros(len(theta_arr))
    
    for l in range (lmax):
        for m_f in range (0,l+1):
            ind_f = hp.Alm.getidx(lmax, l, m_f)
            for m_w in range (0,l+1):
                ind_w = hp.Alm.getidx(lmax, l, m_w)
                ind_D = wigner.Dindex(l, m_w, m_f)
                
                h+=D[ind_D]*field_l [ind_f]*wavelet_l[ind_w]
                
                ind_Dp = wigner.Dindex(l, -m_w, -m_f)
                h+=D[ind_Dp]*field_l [ind_f].conjugate()*wavelet_l[ind_w].conjugate()
                
                
    return h 
                
                
                
                



def gabor_dir(freq, sigma, phi0, theta, phi):
    
    arg=-(theta**2)/(2*sigma*sigma) + 1.j * freq*theta *np.cos (phi-phi0)
    
    g=np.exp(arg)
    g/=2*np.pi*sigma*sigma
    return g


def morlet_dir (f, sigma, phi0, theta, phi, lmax):
    wv=gabor_dir(f, sigma, phi0, theta, phi)
    wvm=gabor_dir(0,sigma, phi0, theta, phi)
    B=integrate_dir(wv,theta,phi)/integrate_dir(wvm,theta,phi)
    mor=wv-B*wvm
    #norm = np.mean(np.abs(mor)**2*np.sin(theta))
    #mor = mor/norm
    print ("hi")
    return mor


 

def morlet_arr_dir (resol, jmax, rmax, nside, lmax) :
    mor=[]
    fl2beam=[]
    r = np.linspace (0,np.pi, rmax+1)[:-1]
    
    npix = hp.nside2npix (nside)
    theta_arr , phi_arr = hp.pix2ang(nside, np.arange (npix))
    for j in range(jmax):
        
        sigma1 = (0.8*resol*2**j)
        freq1=(3.0*np.pi) /(4.0*resol*2**j)
        for phi0 in r:
            print (j, r)
            morf = morlet_dir (freq1, sigma1, phi0, theta_arr, phi_arr, lmax)
            #print ("ok")
            #norm = np.sum((np.abs (morf))**2 * np.sin(theta)*np.pi/theta_bin)
            #morf /= norm
            #print ("okk")
            #mor_l = beam2bl(morf, theta, lmax)
            #norm = np.sqrt(np.sum((np.abs (morf))**2 * np.sin(theta)*np.pi/theta_bin))
            #print ("hello")
            mor.append(morf)

    return mor





def compS1_dir (hmap, mor_arr, jmax, rmax, nside, gaus_l = None , lmax = None , resol = None):

    #if lmax == None :
        #lmax = 3 * nside - 1 
    
    if resol==None :
        resol = hp.nside2resol(nside, arcmin =False )
        
    
    S1=np.zeros((jmax*rmax))
    i1=[]
    
    mapalm=hp.map2alm(hmap, lmax, use_pixel_weights=True)
    
    npix = hp.nside2npix (nside)
    theta_arr , phi_arr = hp.pix2ang(nside, np.arange (npix))
    
    
    
    #mor_l = morlet_arr(resol, jmax, lmax, jbin)
    #gaus_l = gaus_arr(resol, jmax, jbin)
    
    for j in range(jmax*rmax):
        print(j)
        
        #morlet filter 
        
        morf = mor_arr[j]
        morl = hp.map2alm(morf, lmax)
        
        #convolving the map with filter 1
        
        I1 = convolve_dir (mapalm, morl, lmax, theta_arr , phi_arr)

        #nalm=hp.almxfl(mapalm,morl)
        
        #I1=hp.alm2map(nalm, nside, lmax)
        

        #modulus
        I1=np.abs(I1)

        i1.append(I1)
        #Gaussian filter for S1
        
        
        
        if gaus_l == None:
            
            S1map = I1
            
        else:
        
            gausl = gaus_l[j]
            #Convolving with gaussian filter to get S1
            I1alm=hp.map2alm(I1, lmax, use_pixel_weights=True)
            
            nI1alm=hp.almxfl(I1alm, gausl)
        
            S1map=hp.alm2map(nI1alm, nside, lmax)
        
    
        #Averaging the entire map
        S1[j]=np.mean(S1map)
        
    return S1 , i1

def compS2 ( i1 , mor_l , jmax  , nside , gaus_l = None , lmax = None , resol = None ):

    if lmax == None :
        lmax = 3 * nside 
    
    if resol==None :
        resol = hp.nside2resol(nside, arcmin =False )
        
       
    S2=np.zeros((jmax,jmax))
    i2 = []

    for j1 in range(jmax):
        print (j1)
        I1=i1[j1]
        mapalm1=hp.map2alm(I1, lmax ,use_pixel_weights=True)
        i2_tmp = []
        for j2 in range (jmax):
                    
            
            
            #filter 2        
            f2=mor_l[j2]
            
            #Convolving I1 with filter 2
            
            nalm2=hp.almxfl(mapalm1,f2)
            I2=hp.alm2map(nalm2,nside,lmax)
            
            
            #modulus
            I2=np.abs(I2)
            i2_tmp.append (I2)
            
            if gaus_l == None:
            
                S2map = I2
            
            else:
            
                #Gaussian filter to get S2
                gausl2=gaus_l[j2]
            
                #Convolving with gaussian filter to get S2
                I2alm=hp.map2alm(I2,lmax ,use_pixel_weights=True)
                nI2alm=hp.almxfl(I2alm,gausl2)
                S2map=hp.alm2map(nI2alm, nside,lmax)     
                               
            #averaging the entire map
            S2[j1][j2]=np.mean(S2map)
    
        i2.append(i2_tmp)
    
    return S2 , i2







def gaus_arr (resol, jmax, lmax, theta_bin):
    
    theta = np.linspace(0 , np.pi , theta_bin)    
    
    phifilter=[]
    
    for j in range(jmax):
        sigma1 = (0.8*resol*2**j)
        freq1 = (3.0*np.pi) / (4.0*resol*2**j)
        
        phi1 = gabor(0 , sigma1 , theta)
        gab_l = hp.gauss_beam (2 * np.sqrt (2*np.log(2)) * sigma1 ,lmax)
        #gab_lb=hp.beam2bl(phi1.real,theta,lmax=lmax)
        
        phifilter.append(gab_l)
        
    return phifilter


def compS1 (hmap, mor_l, jmax, nside , gaus_l = None , lmax = None , resol = None):

    #if lmax == None :
        #lmax = 3 * nside - 1 
    
    if resol==None :
        resol = hp.nside2resol(nside, arcmin =False )
        
    
    S1=np.zeros((jmax))
    i1=[]
    
    mapalm=hp.map2alm(hmap, lmax, use_pixel_weights=True)
    
    #mor_l = morlet_arr(resol, jmax, lmax, jbin)
    #gaus_l = gaus_arr(resol, jmax, jbin)
    
    for j in range(jmax):
        print(j)
        
        #morlet filter 
        
        morl = mor_l[j]
        
        #convolving the map with filter 1

        nalm=hp.almxfl(mapalm,morl)
        
        I1=hp.alm2map(nalm, nside, lmax)
        

        #modulus
        I1=np.abs(I1)

        i1.append(I1)
        #Gaussian filter for S1
        
        
        
        if gaus_l == None:
            
            S1map = I1
            
        else:
        
            gausl = gaus_l[j]
            #Convolving with gaussian filter to get S1
            I1alm=hp.map2alm(I1, lmax, use_pixel_weights=True)
            
            nI1alm=hp.almxfl(I1alm, gausl)
        
            S1map=hp.alm2map(nI1alm, nside, lmax)
        
    
        #Averaging the entire map
        S1[j]=np.mean(S1map)
        
    return S1 , i1

def compS2 ( i1 , mor_l , jmax  , nside , gaus_l = None , lmax = None , resol = None ):

    if lmax == None :
        lmax = 3 * nside 
    
    if resol==None :
        resol = hp.nside2resol(nside, arcmin =False )
        
       
    S2=np.zeros((jmax,jmax))
    i2 = []

    for j1 in range(jmax):
        print (j1)
        I1=i1[j1]
        mapalm1=hp.map2alm(I1, lmax ,use_pixel_weights=True)
        i2_tmp = []
        for j2 in range (jmax):
                    
            
            
            #filter 2        
            f2=mor_l[j2]
            
            #Convolving I1 with filter 2
            
            nalm2=hp.almxfl(mapalm1,f2)
            I2=hp.alm2map(nalm2,nside,lmax)
            
            
            #modulus
            I2=np.abs(I2)
            i2_tmp.append (I2)
            
            if gaus_l == None:
            
                S2map = I2
            
            else:
            
                #Gaussian filter to get S2
                gausl2=gaus_l[j2]
            
                #Convolving with gaussian filter to get S2
                I2alm=hp.map2alm(I2,lmax ,use_pixel_weights=True)
                nI2alm=hp.almxfl(I2alm,gausl2)
                S2map=hp.alm2map(nI2alm, nside,lmax)     
                               
            #averaging the entire map
            S2[j1][j2]=np.mean(S2map)
    
        i2.append(i2_tmp)
    
    return S2 , i2

def compS3 ( i2  , mor_l, jmax , nside , gaus_l = None , lmax = None , resol = None):

    if lmax == None :
        lmax = 3 * nside 
        
    if resol==None :
        resol = hp.nside2resol(nside, arcmin =False )
        
    
    S3 = np.zeros((jmax,jmax,jmax ))
    
    
    for j1 in range(jmax):
        
        for j2 in range (j1+1 , jmax):
            
            print (j1 , j2)
            I2=i2[j1][j2]
            mapalm2=hp.map2alm(I2, lmax ,use_pixel_weights=True)
            
            for j3 in range (jmax):
                    
                
                #filter 3       
                f3=mor_l[j3]
            
                #Convolving I2 with filter 3
                
                nalm3 = hp.almxfl (mapalm2 , f3)
                I3 = hp.alm2map(nalm3 , nside , lmax)
                
                #modulus
                I3=np.abs(I3)
                
                
                if gaus_l == None:
            
                    S3map = I3
            
                else:

                    #Gaussian filter to get S2
                    gausl3 = gaus_l[j3]
                
                    #Convolving with gaussian filter to get S3
                    I3alm = hp.map2alm (I3 , lmax , use_pixel_weights=True)
                    nI3alm = hp.almxfl(I3alm , gausl3 )
                    S3map = hp.alm2map(nI3alm, nside,lmax)     
                           
                #averaging the entire map
                S3[j1 , j2 , j3] = np.mean(S3map)
    
    return S3 