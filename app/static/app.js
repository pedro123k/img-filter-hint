pages = {
    index: () => {
        const local_states = {}

        const file_selector = document.querySelector("#uploadImg")
        const dropzone = document.querySelector("#drop-zone")
        const upload_btn = document.querySelector("#upload-btn")
        const img_preview = document.querySelector("#img-preview")

        function updatePreview() {
            if (local_states.uploadfile) {
                
                if (local_states.preview_url) {
                    URL.revokeObjectURL(local_states.preview_url)
                }

                local_states.preview_url = URL.createObjectURL(new Blob([local_states.uploadfile]))
                img_preview.src = local_states.preview_url 
            }    
        }

        function init_upload_button() {
            upload_btn.disabled = false
        }

        function upload_img() {

            if (local_states.precontent){
                window.location.href = window.location.origin + "/review" 
                return
            }

            const data = new FormData()
            data.append('file', local_states.uploadfile) 

            fetch("/upload_img", {
                method: "POST",
                body: data 
            }).then(resp => resp.json()).then(data => {
                if (data.ok) {
                    sessionStorage.setItem("upload_uuid", data.uuid)
                    window.location.href = window.location.origin + "/review"
                }
            })
        }

        function preload() {
            
            uuid = sessionStorage.getItem("upload_uuid")

            if (!uuid) 
                return

            fetch(`/api/imgs_cached/${uuid}`).then(resp => {
                if (resp)
                    return resp.blob()
            }).then(blob => {
                if (!blob)
                    return

                local_states.precontent = true
                local_states.preview_url = URL.createObjectURL(blob)
                img_preview.src = local_states.preview_url
                init_upload_button()
            })
        }

        file_selector.addEventListener("change", () => {
            local_states.uploadfile = file_selector.files[0]
            local_states.precontent = false
            updatePreview()

            if (!local_states.next_button_enable) {
                local_states.next_button_enable = true
                init_upload_button()
            } 
        })

        dropzone.addEventListener("dragover", e => {
            e.preventDefault()
            document.querySelector("#dropzone-svg").classList.add("svg-control-is-over")
        })

        dropzone.addEventListener("dragleave", e => {
            e.preventDefault()
            document.querySelector("#dropzone-svg").classList.remove("svg-control-is-over")

        })

        dropzone.addEventListener("drop", e => {
            e.preventDefault()
            document.querySelector("#dropzone-svg").classList.remove("svg-control-is-over")

            const file = e.dataTransfer.files[0]
            
            if (file.type === "image/jpeg") {
                local_states.uploadfile = file
                local_states.precontent = false
                updatePreview()

                if (!local_states.next_button_enable) {
                    local_states.next_button_enable = true
                    init_upload_button()
                }   

            }
        })

        upload_btn.addEventListener("click", () => {
            upload_img()
        })

        preload()
    },

    review: () => {
        const local_states = {}

        const img_loaded = document.querySelector("#img-stored")
        const img_filted = document.querySelector("#img-filtered")
        const form_filter = document.querySelector("#selection-filter")
        const score_label = document.querySelector("#score-value")
        const class_label = document.querySelector("#classification-value")
        const back_btn = document.querySelector("#icon-back")
        const model_detail_btn = document.querySelector("#icon-model-detail")

        function load_stored_img() {
            uuid = sessionStorage.getItem("upload_uuid")

            if (!uuid) 
                return

            fetch(`/api/imgs_cached/${uuid}`).then(resp => resp.blob()).then(blob => {
                local_states.img_url = URL.createObjectURL(blob)
                img_loaded.src = local_states.img_url
            })
        }

        function get_description(label) {
            switch(label) {
                case "OK":
                    return "OK. A imagem está Boa!"
                case "NOISY":
                    return "Ruidoso. Tente um filtro de ruído"
                case "BLURRED":
                    return "Borrado. Tente um filtro de blur"
                case "LOW-LIGHTENED":
                    return "Baixa Luminosidade. Tente um filtro de luminosidade"
                default:
                    return "ERRO! Classificação Inválida!"
            }
        }

        function load_filtered_data(filter) {

            uuid = sessionStorage.getItem("upload_uuid")

            if (!uuid)
                return

            req = (filter === "default") ? `/api/evaluation/${uuid}` : `/api/${filter}/${uuid}`
            
            fetch(req).then(res => {
                if (res.ok)
                    return res.json()
            }).then(json => {
                if (!json)
                    return

                if (filter === "default") 
                    img_filted.src = local_states.img_url         
                
                else 
                    img_filted.src = "data:image/jpeg;base64," + json.content.img_result;
                
                score_label.textContent = (100 * parseFloat(json.content.score)).toFixed(2)
                class_label.textContent = get_description(json.content.label)
            })
        }

        form_filter.addEventListener("change", e => {
            selected = document.querySelector("input[name='filter-selector']:checked")
            load_filtered_data(selected.value)
        })

        back_btn.addEventListener("click", () => window.location.href = window.location.origin)

        model_detail_btn.addEventListener("click", () => window.location.href = window.location.origin + "/model_info")

        load_stored_img()
        load_filtered_data("default")

    },

}

if (document.querySelector("#drop-zone")) {
    pages.index()
}

if (document.querySelector("#img-filtered")) {
    pages.review()
}
