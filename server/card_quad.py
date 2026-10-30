"""Card outline detection and perspective correction for non-Pokemon games.

Uses geometry rather than applying the Pokemon-trained detector to other games.
"""
import cv2
import numpy as np
cv2.setNumThreads(1)


def ordered(points):
    points = np.asarray(points, dtype=np.float32).reshape(4,2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))]
    return np.roll(points,-int(np.argmin(points.sum(axis=1))),axis=0)


def detect(rgb, *, live=False):
    h,w = rgb.shape[:2]
    scale = min(1.,720/max(h,w))
    small = cv2.resize(rgb,(round(w*scale),round(h*scale)))
    gray = cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    # Color edges recover foil/card borders against similarly bright backgrounds.
    saturation = cv2.GaussianBlur(cv2.cvtColor(small,cv2.COLOR_RGB2HSV)[:,:,1],(5,5),0)
    images = [cv2.Canny(gray,35,110),cv2.Canny(gray,70,180),cv2.Canny(saturation,70,180)]
    # Recover low-contrast outer borders against a mostly uniform tabletop.
    lab = cv2.cvtColor(small,cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate((lab[0],lab[-1],lab[:,0],lab[:,-1]))
    background = np.median(border,axis=0)
    cutoff = max(12.,float(np.percentile(np.linalg.norm(border-background,axis=1),90))+6.)
    if cutoff < 60:
        mask = (np.linalg.norm(lab-background,axis=2)>cutoff).astype(np.uint8)*255
        images.append(mask)
    candidates=[]
    for edges in images:
        edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        outlines = list(contours) + [cv2.convexHull(c) for c in contours]
        for contour, epsilon in ((c,e) for c in outlines for e in (.015,.025)):
            area=abs(cv2.contourArea(contour))/scale**2
            if area < .035*w*h: continue
            approx=cv2.approxPolyDP(contour,epsilon*cv2.arcLength(contour,True),True)
            if len(approx)!=4 or not cv2.isContourConvex(approx):continue
            quad=ordered(approx.reshape(4,2)/scale)
            sides=np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)
            cw=(sides[0]+sides[2])/2;ch=(sides[1]+sides[3])/2
            ratio=min(cw,ch)/max(cw,ch)
            if not .53<=ratio<=.86 or sides.min()<35:continue
            # Reject extreme perspective distortions. Nested borders are compared by recognition.
            if min(sides[0],sides[2])/max(sides[0],sides[2])<.55:continue
            if min(sides[1],sides[3])/max(sides[1],sides[3])<.55:continue
            xy=[float(quad[:,0].min()),float(quad[:,1].min()),float(quad[:,0].max()),float(quad[:,1].max())]
            candidates.append({'xyxy':xy,'quad':quad.tolist(),'conf':1.,'detector':'card-quad','area':area})
    # Keep distinct nested borders: the largest rectangle may be a webpage/photo.
    kept=[]
    for box in sorted(candidates,key=lambda x:x['area'],reverse=True):
        x1,y1,x2,y2=box['xyxy']
        duplicate=False
        for prior in kept:
            a,b,c,d=prior['xyxy']
            intersection=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
            area=(x2-x1)*(y2-y1); prior_area=(c-a)*(d-b)
            union=area+prior_area-intersection
            if intersection/max(1,union)>.88:duplicate=True;break
        if not duplicate:
            box.pop('area');kept.append(box)
    # A tightly cropped gallery image already has card boundaries.
    if not live and .60<=min(w,h)/max(w,h)<=.80:
        full={'xyxy':[0.,0.,float(w),float(h)],'conf':1.,'detector':'card-frame'}
        if not kept:
            return [full]
    return kept[:12]


def crop(rgb, box):
    if 'quad' not in box:
        x1,y1,x2,y2=map(lambda x:int(round(x)),box['xyxy'])
        return rgb[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
    p=ordered(box['quad']);sides=np.linalg.norm(p-np.roll(p,-1,axis=0),axis=1)
    w=max(8,round(max(sides[0],sides[2])));h=max(8,round(max(sides[1],sides[3])))
    dest=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32)
    matrix=cv2.getPerspectiveTransform(p,dest)
    return cv2.warpPerspective(rgb,matrix,(w,h))
