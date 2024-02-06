function initImgClick() {
    let containers = document.getElementsByClassName("hri-images-container");
    Array.from(containers).forEach((container) => {
        let imgs = container.getElementsByTagName("img");
        Array.from(imgs).forEach((img)=>{
            img.onpointerover = ()=>{
                img.style.cursor = "pointer";
            }
            img.onpointerout = ()=>{
                img.style.cursor = "default";
            }
            img.onclick = ()=>{
                window.open(img.src,"_blank");
            }
        })
    });
}